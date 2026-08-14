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
    if (!values || Object.entries(values).length == 0) {
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
    let labels = model.get("dimension_labels") ?? {};

    noUiSlider.cssClasses.target += " xdggs-slider";

    el.classList.add("xdggs-dimension-sliders");
    const digits = Math.max(...Object.values(dimensions)).toString().length;
    const labelChars = Math.max(
      ...Object.values(labels).map((labelList) =>
        Math.max(...labelList.map((label) => label.length)),
      ),
    );

    el.style.setProperty(
      "--xdggs-dimension-value-width",
      `${Math.max(digits, labelChars)}ch`,
    );

    const sliders = Object.fromEntries(
      Object.keys(dimensions).map((name) => {
        const nameLabel = document.createElement("span");
        const slider = document.createElement("div");
        const valueLabel = document.createElement("span");

        el.appendChild(nameLabel);
        el.appendChild(slider);
        el.appendChild(valueLabel);

        /* name label */
        nameLabel.innerText = name;

        /* slider */
        const value = values[name];
        slider.id = `xdggs-slider-${name}`;
        slider.classList.add("xdggs-slider-container");

        const sliderWidget = document.createElement("div");
        slider.appendChild(sliderWidget);

        const max = dimensions[name] - 1;

        noUiSlider.create(sliderWidget, {
          start: values[name],
          step: 1,
          range: { min: 0, max: dimensions[name] - 1 },
          connect: "lower",
        });
        setDisabled(sliderWidget, !enabled[name]);

        sliderWidget.noUiSlider.on("update.xdggs", ([formatted_value]) => {
          const value = parseInt(formatted_value);

          let label;
          if (labels[name] !== undefined) {
            label = labels[name][value];
          } else {
            label = value;
          }

          valueLabel.innerText = label;

          const new_values = { ...model.get("dimension_values") };
          new_values[name] = value;
          model.set("dimension_values", new_values);
          model.save_changes();
        });

        /* value label */
        valueLabel.innerText = value;

        return [name, sliderWidget];
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
