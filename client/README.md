# moshi-client

Frontend for the demo.

## Run the client

- Node is required, I recommend using [NVM](https://github.com/nvm-sh/nvm) to help you manage your node version and make sure you're on the recommended version for this project. If you do so run `nvm use`.
- Generate a public/private key pair named `cert.pem` and `key.pem`, then place both files at the root of this package.
- Create `.env.local` and add:
  - `VITE_QUEUE_API_PATH=/api`
  - `VITE_QUEUE_API_URL=https://localhost:8998` (or your worker URL)
  - Optional: `VITE_DEV_HTTPS=false` to run plain HTTP in development
- Before running the project for the time or after dependencies update use `npm install`
- To run the project use `npm run dev`
- To build the project use `npm run build`

Vite runs as HTTPS by default and reverse-proxies both HTTP and WebSocket `/api/*` traffic to `VITE_QUEUE_API_URL`.

## Skipping the queue
To skip the queue for standalone use, once the project is running go to `/?worker_addr={WORKER_ADDR}` where `WORKER_ADDR` is your worker instance address.
For example : `https://localhost:5173/?worker_addr=0.0.0.0:8088`

## License

The present code is provided under the MIT license.
