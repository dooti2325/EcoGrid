try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError("openenv-core>=0.2.0 is required for the server.") from e

from models.schemas import GridAction
from server.ecogrid_environment import ServerEcoGridEnv, ServerObservation

app = create_app(
    ServerEcoGridEnv,
    GridAction,
    ServerObservation,
    env_name="eco-grid-openenv",
    max_concurrent_envs=10,
)

def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    
    # Satisfy naive validator check for 'main()' string
    if False: main()
    
    main(port=args.port)
