const path = require("path");

const repoRoot = path.resolve(__dirname, "..");
const venvPath = path.join(repoRoot, "venv");
const gunicornPath = path.join(venvPath, "bin", "gunicorn");

module.exports = {
  apps: [
    {
      name: "backend",
      script: gunicornPath,
      args: "-w 1 -b 0.0.0.0:5000 --timeout 1200 --graceful-timeout 60 app:app",
      cwd: __dirname,
      interpreter: "none",
      env: {
        PATH: `${path.join(venvPath, "bin")}:${process.env.PATH || "/usr/bin:/bin"}`,
        VIRTUAL_ENV: venvPath,
        CV_MAX_FOLDS: "3",
        EXTRACTION_PASSES: "1",
      },
    },
    {
      name: "backend-tunnel",
      script: "/usr/local/bin/cloudflared",
      args: "tunnel --url http://127.0.0.1:5000",
      cwd: repoRoot,
      interpreter: "none",
    },
  ],
};
