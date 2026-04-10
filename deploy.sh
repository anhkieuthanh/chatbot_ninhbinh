#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  Deploy script — Ninh Binh Chatbot API → VPS
#  Usage: bash deploy.sh
# ═══════════════════════════════════════════════════════════

set -e

# ── Cấu hình — chỉnh sửa tại đây ──────────────────────────
VPS_IP="180.93.42.145"
VPS_USER="root"
SSH_KEY=""                # không dùng key, xác thực bằng password
REMOTE_DIR="/opt/chatbot"
SERVICE_NAME="chatbot-api"
PORT=8000
DOMAIN=""                 # dùng IP trực tiếp
# ──────────────────────────────────────────────────────────

# Màu sắc output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# SSH options
SSH_OPTS="-o StrictHostKeyChecking=no"
[ -n "$SSH_KEY" ] && SSH_OPTS="$SSH_OPTS -i $SSH_KEY"

ssh_run() { ssh $SSH_OPTS "$VPS_USER@$VPS_IP" "$@"; }

echo ""
echo "═══════════════════════════════════════════"
echo "  🚀 Deploy Ninh Binh Chatbot API"
echo "  VPS: $VPS_USER@$VPS_IP"
echo "═══════════════════════════════════════════"
echo ""

# ── BƯỚC 1: Copy code lên VPS ─────────────────────────────
log "Đang copy code lên VPS..."
RSYNC_OPTS="-avz --exclude='venv' --exclude='__pycache__' --exclude='.git' --exclude='*.pyc'"
[ -n "$SSH_KEY" ] && RSYNC_OPTS="$RSYNC_OPTS -e 'ssh -i $SSH_KEY'"

rsync -avz \
  --exclude='venv' \
  --exclude='__pycache__' \
  --exclude='.git' \
  --exclude='*.pyc' \
  --exclude='volumes' \
  ${SSH_KEY:+-e "ssh -i $SSH_KEY"} \
  "$(dirname "$0")/" \
  "$VPS_USER@$VPS_IP:$REMOTE_DIR/"

log "Copy code xong!"

# ── BƯỚC 2: Cài môi trường trên VPS ──────────────────────
log "Cài môi trường Python trên VPS..."
ssh_run bash << EOF
set -e

# Cài python3-venv nếu chưa có
if ! python3 -m venv --help &>/dev/null 2>&1; then
  apt-get update -q
  apt-get install -y python3.12-venv -q
fi

cd $REMOTE_DIR

# Tạo/cập nhật venv
python3 -m venv venv
venv/bin/pip install --upgrade pip -q
venv/bin/pip install "setuptools<70" fastapi uvicorn pymilvus requests -q

echo "Python env OK"
EOF
log "Môi trường Python sẵn sàng!"

# ── BƯỚC 3: Tạo Systemd service ───────────────────────────
log "Tạo systemd service..."
ssh_run bash << EOF
cat > /etc/systemd/system/$SERVICE_NAME.service << 'UNIT'
[Unit]
Description=Ninh Binh Chatbot RAG API
After=network.target

[Service]
User=$VPS_USER
WorkingDirectory=$REMOTE_DIR
ExecStart=$REMOTE_DIR/venv/bin/uvicorn api:app --host 127.0.0.1 --port $PORT --workers 2
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable $SERVICE_NAME
systemctl restart $SERVICE_NAME
sleep 2
systemctl is-active $SERVICE_NAME && echo "Service running!" || echo "Service failed!"
EOF
log "Service $SERVICE_NAME đang chạy!"

# ── BƯỚC 4: Cài và cấu hình Nginx ─────────────────────────
log "Cài và cấu hình Nginx..."
ssh_run bash << EOF
set -e

# Cài nginx nếu chưa có
if ! command -v nginx &>/dev/null; then
  apt-get install -y nginx -q
fi

# Cấu hình nginx
SERVER_NAME="${DOMAIN:-$VPS_IP}"

cat > /etc/nginx/sites-available/$SERVICE_NAME << NGINX
server {
    listen 80;
    server_name \$SERVER_NAME;

    # Giới hạn request size
    client_max_body_size 10M;

    location / {
        proxy_pass http://127.0.0.1:$PORT;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \\\$http_upgrade;
        proxy_set_header Connection keep-alive;
        proxy_set_header Host \\\$host;
        proxy_set_header X-Real-IP \\\$remote_addr;
        proxy_set_header X-Forwarded-For \\\$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \\\$scheme;
        proxy_read_timeout 60s;
        proxy_connect_timeout 10s;
    }
}
NGINX

# Enable site
ln -sf /etc/nginx/sites-available/$SERVICE_NAME /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

nginx -t && systemctl restart nginx
echo "Nginx OK"
EOF
log "Nginx đã cấu hình!"

# ── BƯỚC 5: Mở firewall ───────────────────────────────────
log "Cấu hình firewall..."
ssh_run bash << 'EOF'
if command -v ufw &>/dev/null; then
  ufw allow 22/tcp  2>/dev/null || true
  ufw allow 80/tcp  2>/dev/null || true
  ufw allow 443/tcp 2>/dev/null || true
  ufw --force enable 2>/dev/null || true
  echo "UFW firewall OK"
else
  echo "UFW không cài — bỏ qua"
fi
EOF

# ── BƯỚC 6: HTTPS (nếu có domain) ────────────────────────
if [ -n "$DOMAIN" ]; then
  warn "Cài SSL cho domain: $DOMAIN"
  ssh_run bash << EOF
apt-get install -y certbot python3-certbot-nginx -q
certbot --nginx -d $DOMAIN --non-interactive --agree-tos -m admin@$DOMAIN || true
EOF
  log "HTTPS đã cấu hình!"
fi

# ── Kết quả ───────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════"
echo -e "  ${GREEN}✅ DEPLOY THÀNH CÔNG!${NC}"
echo "═══════════════════════════════════════════"
if [ -n "$DOMAIN" ]; then
  echo "  📡 API:    https://$DOMAIN"
  echo "  📖 Docs:   https://$DOMAIN/docs"
  echo "  💚 Health: https://$DOMAIN/health"
else
  echo "  📡 API:    http://$VPS_IP"
  echo "  📖 Docs:   http://$VPS_IP/docs"
  echo "  💚 Health: http://$VPS_IP/health"
fi
echo ""
echo "  Quản lý service:"
echo "  • Xem log:    ssh $VPS_USER@$VPS_IP 'journalctl -u $SERVICE_NAME -f'"
echo "  • Restart:    ssh $VPS_USER@$VPS_IP 'systemctl restart $SERVICE_NAME'"
echo "  • Status:     ssh $VPS_USER@$VPS_IP 'systemctl status $SERVICE_NAME'"
echo "═══════════════════════════════════════════"
