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

`GraphController` is an internal composition host. Feature methods declare an
explicit typed `this` parameter and share one controller instance, graph, renderer,
and navigation record. Methods are installed on its prototype with ordinary class
descriptors; duplicate names fail at initialization. Type-only imports keep feature
modules from creating runtime cycles. The reader uses the small `ChronicleGraph`
interface rather than the internal state.

Feature modules share one navigation record across panels. Source-size checks
keep maintained modules within the 2,000-line ceiling.

The [construction guide](../../../docs/BUILD_PROCESS.md) connects the interface
to its source-linked claims, embeddings, categories and graph snapshots.
