# Database Setup

The project uses **PostgreSQL + PostGIS**, run locally as a container defined in
[`compose.yaml`](../compose.yaml). Connection settings match the `DATABASE_URL` hardcoded in
`src/data_utils.py`:

```
postgresql://postgres:password@localhost:5432/seaice
```

| Setting  | Value             |
|----------|-------------------|
| Host     | `localhost`       |
| Port     | `5432`            |
| Database | `seaice`          |
| User     | `postgres`        |
| Password | `password`        |
| Image    | `postgis/postgis:16-3.4` (PostGIS 3.4, PostgreSQL 16) |
| Volume   | `seaice-pgdata` (named, persistent) |

## Quick start

```bash
podman compose up -d        # or: docker compose up -d
podman compose ps           # check status (should be "Up")
podman compose logs -f db   # follow logs
podman compose down         # stop (keeps data)
podman compose down -v      # stop AND delete the data volume (fresh start)
```

The PostGIS extension is enabled automatically by the image on first initialization, so
`postgis_version()` works out of the box. The project's `ice_extent_pan_arctic_daily`,
`ice_extent_regional_daily`, and `ice_extent_climatology` tables are **not** created by compose —
they are created by `notebooks/01b_data_ingestion_nsidc.ipynb` when NSIDC data is ingested.

## One-time Podman setup (rootless)

`podman compose` delegates to the docker-compose provider, which talks to the Podman API socket.
Enable it once per user (no sudo required):

```bash
systemctl --user enable --now podman.socket
```

If the compose provider can't find the socket, point it explicitly:

```bash
export DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock
```

To have the container come back after a reboot, either enable Podman's user socket + lingering,
or rely on the `restart: unless-stopped` policy in `compose.yaml` (which restarts the container
as long as the Podman service is running).

## Verifying the connection

```bash
uv run python -c "import sqlalchemy; from src.data_utils import DATABASE_URL; \
  print(sqlalchemy.create_engine(DATABASE_URL).connect().execute(\
  sqlalchemy.text('select postgis_version()')).scalar())"
```
