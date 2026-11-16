document.addEventListener("click", (event) => {
  if (!event.target.closest("#download-record")) return;
  const json = document.getElementById("run-record").textContent;
  const url = URL.createObjectURL(new Blob([json], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "grelu-analysis.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
