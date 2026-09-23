/** Load graph features with an explicit DOM/worker boundary and no server or GPU. */
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const ts = require("typescript");

class ElementFixture extends EventTarget {
  constructor(id = "") {
    super();
    this.id = id;
    this.value = "";
    this.hidden = false;
    this.disabled = false;
    this.checked = false;
    this.dataset = {};
    this.children = [];
    this.classList = { add() {}, remove() {}, toggle() {} };
  }
  append(...nodes) { this.children.push(...nodes); }
  appendChild(node) { this.children.push(node); return node; }
  replaceChildren(...nodes) { this.children = nodes; }
  querySelectorAll() { return []; }
  remove() {}
}

function graphFixture() {
  const elements = new Map();
  const workers = [];
  const document = {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, new ElementFixture(id));
      return elements.get(id);
    },
    createElement: () => new ElementFixture(),
  };
  class WorkerFixture {
    constructor() { this.listeners = new Map(); workers.push(this); }
    addEventListener(name, handler) { this.listeners.set(name, handler); }
    postMessage(message, transfer) { this.message = message; this.transfer = transfer; }
    terminate() { this.terminated = true; }
    reply(data) { this.listeners.get("message")({ data }); }
  }
  const unsupported = () => { throw new Error("Rendering is outside the graph-state fixture"); };
  const context = vm.createContext({
    document,
    console,
    Map, Set, URLSearchParams, AbortController, DOMException,
    performance: { now: () => 100 },
    window: {
      setTimeout, clearTimeout, requestAnimationFrame: () => 1,
      cancelAnimationFrame() {},
    },
    fetch: () => { throw new Error("Unexpected network request in graph-state fixture"); },
    HTMLElement: ElementFixture,
    HTMLInputElement: ElementFixture,
    HTMLDetailsElement: ElementFixture,
  });
  const cache = new Map();
  const root = path.resolve(__dirname, "../frontend");
  function load(file) {
    file = path.resolve(file);
    if (cache.has(file)) return cache.get(file).exports;
    const module = { exports: {} };
    cache.set(file, module);
    const source = fs.readFileSync(file, "utf8");
    const code = ts.transpileModule(source, {
      compilerOptions: {
        target: ts.ScriptTarget.ES2022,
        module: ts.ModuleKind.CommonJS,
        esModuleInterop: true,
      },
      fileName: file,
    }).outputText;
    function resolve(specifier) {
      if (specifier.endsWith("?worker")) return WorkerFixture;
      if (specifier === "sigma/rendering") return {
        createEdgeArrowProgram: () => class {}, EdgeLineProgram: class {}, NodePointProgram: class {},
      };
      if (specifier === "sigma/utils") return { animateNodes: unsupported };
      if (specifier === "sigma" || specifier === "graphology-layout-forceatlas2/worker") {
        return unsupported;
      }
      if (specifier.startsWith(".")) return load(path.resolve(path.dirname(file), specifier + ".ts"));
      return require(specifier);
    }
    const run = vm.runInContext(`(function(require, module, exports) {\n${code}\n})`, context, { filename: file });
    run(resolve, module, module.exports);
    return module.exports;
  }
  const { GraphController } = load(path.join(root, "graph/controller.ts"));
  const { el } = load(path.join(root, "graph/dom.ts"));
  return { controller: new GraphController(), GraphController, el, workers };
}

module.exports = { graphFixture };
