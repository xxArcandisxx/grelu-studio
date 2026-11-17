const element = (id) => document.getElementById(id);
const form = element("compare");
let result;
let generation = 0;
const format = (number) => Number(number).toPrecision(5);

function clearResult() {
  result = undefined;
  element("output").hidden = true;
  element("download").hidden = true;
  element("empty").hidden = false;
}

async function getJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || (response.status === 413
    ? "The sequence upload is too large." : "The server could not complete the request."));
  return data;
}

for (const name of ["reference", "alternate"]) {
  element(name + "-file").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    event.target.value = "";
    try {
      if (file.size > 300000) throw new Error("Choose a FASTA file under 300 KB.");
      element(name).value = await file.text();
      element(name).dispatchEvent(new Event("input", { bubbles: true }));
    } catch (error) {
      element("status").textContent = error.message;
    }
  });
}
form.addEventListener("input", () => {
  generation += 1;
  clearResult();
  element("status").textContent = "";
});

function renderPlot(reference, alternate) {
  const plot = element("plot");
  plot.replaceChildren();
  const values = [...reference, ...alternate];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const y = (value) => max === min ? 95 : 177 - 160 * (value - min) / span;
  const add = (type, attributes, text) => {
    const node = document.createElementNS("http://www.w3.org/2000/svg", type);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== undefined) node.textContent = text;
    plot.append(node);
  };
  for (const value of new Set([min, (min + max) / 2, max])) {
    add("line", { x1: 90, y1: y(value), x2: 590, y2: y(value), stroke: "#dce4ec" });
    add("text", { x: 5, y: y(value) + 4, fill: "#526176", "font-size": 12 }, format(value));
  }
  [reference, alternate].forEach((series, index) => {
    const stroke = index ? "#a65027" : "#1c6387";
    if (series.length === 1) {
      add("circle", { cx: 340 + 8 * index, cy: y(series[0]), r: 4, fill: stroke });
      return;
    }
    const points = series.map((value, i) => [90 + 500 * i / (series.length - 1), y(value)].join(","));
    add("polyline", { points: points.join(" "), fill: "none", stroke,
      "stroke-width": 2, "stroke-dasharray": index ? "5 4" : "none" });
  });
  add("text", { x: 340, y: 202, "text-anchor": "middle", fill: "#526176", "font-size": 12 },
    reference.length + " displayed output bins");
}

function showTrack() {
  const key = element("track").value;
  const [reference, alternate] = result.results.predictions;
  element("ref-score").textContent = format(reference.scores[key]);
  element("alt-score").textContent = format(alternate.scores[key]);
  element("delta").textContent = format(alternate.scores[key] - reference.scores[key]);
  renderPlot(reference.tracks[key], alternate.tracks[key]);
  const diagnostics = [
    ["GC content", ...[reference, alternate].map((row) => (row.gc_content * 100).toFixed(1) + "%")],
    ["Reverse-complement score", ...[reference, alternate].map((row) => format(row.reverse_complement_scores[key]))],
    ["Absolute forward / reverse gap", ...[reference, alternate].map((row) =>
      format(Math.abs(row.scores[key] - row.reverse_complement_scores[key])))],
  ];
  element("diagnostics").replaceChildren(...diagnostics.map((values) => {
    const row = document.createElement("tr");
    values.forEach((value, index) => {
      const cell = document.createElement(index ? "td" : "th");
      if (!index) cell.scope = "row";
      cell.textContent = value;
      row.append(cell);
    });
    return row;
  }));
}
element("track").addEventListener("change", showTrack);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const current = ++generation;
  clearResult();
  element("run").disabled = true;
  element("status").textContent = "Running prediction…";
  try {
    const data = await getJson("/api/compare", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(Object.fromEntries(["model", "reference", "alternate"].map((key) => [key, element(key).value]))),
    });
    if (current !== generation) return;
    result = data;
    element("comparison-saved").hidden = !data.saved_run_id;
    if (data.saved_run_id) {
      element("comparison-open").href = "/experiments/" + data.saved_run_id;
      element("comparison-report").href = "/experiments/" + data.saved_run_id + "/report.html";
    }
    element("track").replaceChildren(...data.model.tasks.map((name) => new Option(name, name)));
    const actual = data.demo ? "Demo CNN · random weights" : data.model.provenance.model_id;
    element("model-note").textContent = (data.model.catalog_selection ? "Selected " + data.selected_model + " · " : "") +
      "Model used " + actual;
    element("context").textContent = "Position " + data.inputs.position + " · " +
      data.inputs.substitution + " · " + data.inputs.reference.length + " bp · " + data.runtime_s + " s";
    const notes = ["Reverse-complement scores check how predictions change when the DNA window is read on the opposite strand. They are a consistency check.",
      ...data.evidence.map((entry) => entry.name + " — " + entry.message), ...data.warnings];
    element("evidence").replaceChildren(...notes.map((note) => {
      const li = document.createElement("li");
      li.textContent = note;
      return li;
    }));
    showTrack();
    element("empty").hidden = true;
    element("output").hidden = false;
    element("download").hidden = false;
    element("status").textContent = "Comparison complete.";
  } catch (error) {
    if (current === generation) element("status").textContent = error.message;
  } finally {
    element("run").disabled = false;
  }
});

element("download").addEventListener("click", () => {
  if (!result) return;
  const track = element("track").value;
  const [ref, alt] = result.results.predictions;
  const url = URL.createObjectURL(new Blob([JSON.stringify({
    ...result, selected_track: track, selected_delta: alt.scores[track] - ref.scores[track],
  }, null, 2)], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "grelu-comparison.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
