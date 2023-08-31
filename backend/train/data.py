from dataclasses import dataclass


# Always change together with Android `HttpClient.ServerData`
# & Dart `backend_client.ServerData`.
@dataclass
class ServerData:
    status: str
    session_id: int | None
    port: int | None
