const scanForm = document.getElementById("scan-form");
const runButton = document.getElementById("scan-button");
const statusMessage = document.getElementById("scan-status");
const results = document.getElementById("scan-results");

scanForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = new FormData(scanForm);
  runButton.disabled = true;
  scanForm.elements.locus.disabled = true;
  runButton.textContent = "Running scan…";
  statusMessage.textContent = "Scoring 24 substitutions. The first run can take a minute or more.";
  results.setAttribute("aria-busy", "true");
  results.querySelectorAll(":scope > :not(h2)").forEach((node) => node.remove());
  try {
    const response = await fetch(scanForm.action || location.href, { method: "POST", body: input });
    const page = new DOMParser().parseFromString(await response.text(), "text/html");
    if (!response.ok) {
      const error = page.getElementById("status");
      throw new Error(error?.textContent || "The scan failed on the server.");
    }
    const rendered = page.getElementById("scan-results");
    if (!rendered?.querySelector("#run-record")) throw new Error("The server returned no scan results.");
    results.replaceChildren(...rendered.childNodes);
    statusMessage.textContent = "Scan complete. The heatmap is ready.";
  } catch (error) {
    statusMessage.textContent = error instanceof TypeError
      ? "Could not reach the scan server. Refresh the page and try again."
      : error.message;
  } finally {
    runButton.disabled = false;
    scanForm.elements.locus.disabled = false;
    runButton.textContent = "Run live scan";
    results.removeAttribute("aria-busy");
  }
});
