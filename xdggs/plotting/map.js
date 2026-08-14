export default {
  async render({ model, el, signal, host }) {
    el.classList.add("xdggs-map");

    const map = await host.getWidget(model.get("map"));
    const control = await host.getWidget(model.get("control"));

    let mapElement = document.createElement("div");
    let controlElement = document.createElement("div");

    el.appendChild(mapElement);
    el.appendChild(controlElement);

    await map.render({ model, el: mapElement, signal, host });
    await control.render({ model, el: controlElement, signal, host });
  },
};
