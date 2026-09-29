# Graph interface modules

[`../graph.ts`](../graph.ts) binds page controls and publishes `window.ChronicleGraph`.
Vite builds `static/build/graph.js` and manages the layout-worker assets.

| Responsibility | Modules |
|---|---|
| Shared state and composition | `controller.ts`, `types.ts`, `navigation_state.ts` |
| Page controls and workspace lifecycle | `dom.ts`, `page_navigation.ts`, `workspace.ts` |
| Graph construction and rendering | `graph_layout.ts`, `renderer_lifecycle.ts`, `camera.ts` |
| Thematic graph | `thematic_model.ts`, `thematic_rendering.ts`, `thematic_navigation.ts`, `thematic_evidence.ts` |
| Corpus and atlas views | `corpus_radial_model.ts`, `corpus_radial_view.ts`, `atlas_view.ts` |
| Filters, inspection, and similarity | `category_filters.ts`, `filtered_tags.ts`, `selection_inspector.ts`, `similar_patterns.ts` |
| Shared calculations and requests | `utilities.ts` |

All feature modules operate on one shared `GraphController` instance, which holds
the graph, the renderer and the navigation state. The reader page interacts with the
graph only through the `ChronicleGraph` interface.
