export default {
  initialize({ model, signal }) {},
  async render({ model, el, signal, host }) {
    el.classList.add("xdggs-control-panel");

    const variableChooser = await host.getWidget(model.get("variable_chooser"));
    const sliders = await host.getWidget(model.get("dimension_sliders"));
    const colorbar = await host.getWidget(model.get("colorbar"));

    let variableElement = document.createElement("div");
    let slidersElement = document.createElement("div");
    let colorbarElement = document.createElement("div");
    el.appendChild(variableElement);
    el.appendChild(slidersElement);
    el.appendChild(colorbarElement);

    await variableChooser.render({ el: variableElement, model, signal, host });
    await sliders.render({ el: slidersElement, model, signal, host });
    await colorbar.render({ el: colorbarElement, model, signal, host });
  },
};
