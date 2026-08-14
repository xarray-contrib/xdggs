import * as noUiSlider from "https://esm.sh/nouislider@15.8.1";

function setDisabled(slider, value) {
  if (value) {
    slider.noUiSlider.disable();
  } else {
    slider.noUiSlider.enable();
  }
}

export default {
  initialize({ model, signal }) {
    const dimensions = model.get("dimensions");

    let enabled = model.get("dimension_available");
    if (!enabled || Object.entries(enabled).length == 0) {
      const enabled = Object.fromEntries(
        Object.keys(dimensions).map((name) => {
          return [name, true];
        }),
      );
      model.set("dimension_available", enabled);
    }

    let values = model.get("dimension_values");
    if (!values || Object.entries(enabled).length == 0) {
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

    noUiSlider.cssClasses.target += " xdggs-slider";

    el.classList.add("xdggs-dimension-sliders");
    const digits = Math.max(...Object.values(dimensions)).toString().length;
    el.style.setProperty("--xdggs-dimension-value-width", `${digits}ch`);

    const sliders = Object.fromEntries(
      Object.keys(dimensions).map((name) => {
        const label = document.createElement("span");
        const slider = document.createElement("div");
        const valueLabel = document.createElement("span");

        el.appendChild(label);
        el.appendChild(slider);
        el.appendChild(valueLabel);

        label.innerText = name;

        const value = values[name];

        slider.id = `xdggs-slider-${name}`;

        const max = dimensions[name] - 1;
        console.log("range:", 0, max);
        noUiSlider.create(slider, {
          start: values[name],
          step: 1,
          range: { min: 0, max: dimensions[name] - 1 },
          connect: "lower",
        });
        setDisabled(slider, !enabled[name]);

        valueLabel.innerText = value;

        slider.noUiSlider.on("update.xdggs", ([formatted_value]) => {
          const value = parseInt(formatted_value);
          valueLabel.innerText = value;

          const new_values = { ...model.get("dimension_values") };
          new_values[name] = value;
          model.set("dimension_values", new_values);
          model.save_changes();
        });

        el.appendChild(label);
        el.appendChild(slider);
        el.appendChild(valueLabel);

        return [name, slider];
      }),
    );

    model.on("change:dimension_available", () => {
      let enabled = model.get("dimension_available");

      Object.entries(enabled).forEach(([name, value]) => {
        setDisabled(sliders[name], !enabled[name]);
      });
    });
  },
};
