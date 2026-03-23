module.exports = {
  apps: [{
    name: 'backend',
    script: '/home/kovm23/BP/microsoft-edu-ai-hackathon-2025/venv/bin/gunicorn',
    args: '-w 1 -b 0.0.0.0:5000 --timeout 600 app:app',
    cwd: '/home/kovm23/BP/microsoft-edu-ai-hackathon-2025/backend',
    interpreter: 'none',
    env: {
      PATH: '/home/kovm23/BP/microsoft-edu-ai-hackathon-2025/venv/bin:/usr/bin:/bin',
      VIRTUAL_ENV: '/home/kovm23/BP/microsoft-edu-ai-hackathon-2025/venv',
    }
  }]
};
