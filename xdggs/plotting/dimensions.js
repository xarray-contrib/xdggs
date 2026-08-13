export default {
  initialize({ model, signal }) {
    const dimensions = model.get("dimensions");

    let enabled = model.get("dimension_available");
    if (!enabled) {
      const enabled = Object.fromEntries(
        Object.keys(dimensions).map((name) => {
          return [name, true];
        }),
      );
      model.set("dimension_available", enabled);
    }

    let values = model.get("dimension_values");
    if (!values) {
      const values = Object.fromEntries(
        Object.keys(dimensions).map((name) => {
          return [name, 0];
        }),
      );
      model.set("dimension_values", values);
    }

    model.save_changes();
  },
  async render({ model, el, signal, host }) {
    let dimensions = model.get("dimensions");
    let enabled = model.get("dimension_available");
    let values = model.get("dimension_values");

    el.style.setProperty("display", "grid");
    el.style.setProperty("padding", "5px");
    el.style.setProperty("column-gap", "10px");

    const sliders = Object.fromEntries(
      Object.keys(dimensions).map((name) => {
        const label = document.createElement("span");
        label.innerText = name;

        const value = values[name];

        let input = document.createElement("input");
        input.setAttribute("type", "range");
        input.setAttribute("min", 0);
        input.setAttribute("value", values[name]);
        input.setAttribute("max", dimensions[name]);

        input.disabled = !enabled[name];

        let valueLabel = document.createElement("span");
        valueLabel.innerText = value;

        input.oninput = () => {
          valueLabel.innerText = input.value;

          const new_values = { ...model.get("dimension_values") };
          new_values[name] = Number(input.value);
          model.set("dimension_values", new_values);
          model.save_changes();
        };

        el.appendChild(label);
        el.appendChild(input);
        el.appendChild(valueLabel);

        return [name, input];
      }),
    );

    model.on("change:dimension_available", () => {
      let enabled = model.get("dimension_available");

      Object.entries(enabled).forEach(([name, value]) => {
        sliders[name].disabled = !value;
      });
    });

    el.style.setProperty(
      "grid-template-columns",
      "max-content max-content max-content",
    );
  },
};
