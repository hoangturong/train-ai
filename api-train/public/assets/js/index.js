const root = document.getElementById("root");
window.addEventListener("DOMContentLoaded", async () => {
  const getData = await (async () => {
    const res = await fetch("/api");
    return await res.json();
  })();
  await (async (data) => {
    const html = Object.keys(data).map((item) => {
      const el = document.createElement("div");
      return el.classList.add("card"), (el.innerHTML = data[item]), el;
    });
    root.append(...html);
  })(getData);
});
