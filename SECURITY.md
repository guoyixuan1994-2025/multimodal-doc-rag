# Security Notes

- Never commit `.env` or `.env.docker`.
- Rotate any API key that has previously been committed or shared.
- Set a strong `APP_API_KEY` before exposing the API outside localhost.
- Put the service behind HTTPS and a reverse proxy in public environments.
- Add upload size, file type, rate, and storage quotas before production use.
- Disable or restrict server-side path ingestion endpoints in untrusted environments.

