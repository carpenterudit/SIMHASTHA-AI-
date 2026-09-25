# Architecture

The dashboard is a React single-page interface composed from independently reusable operational widgets. `src/data/mockData.js` is the sole present data source. Components do not contain operational values; they receive data as props where applicable.

`src/services/api.js` is the boundary for backend communication. As FastAPI endpoints are added, dashboard data loading can move from the mock module to typed API calls without requiring each visualization component to change.

The CCTV and map panels are deliberately isolated components. A real video stream player can replace the internal placeholder in `CctvPanel`, and Leaflet or Mapbox can replace the simulated surface in `UjjainMap` while retaining their parent dashboard contracts.

The backend currently exposes only `/api/health`. Routes for future detection, tracking, crowd, congestion, missing-person, emergency, medical, traffic, and drone systems should be implemented as dedicated route modules, with orchestration/business logic in `backend/services` and request/response models in `backend/models`.
