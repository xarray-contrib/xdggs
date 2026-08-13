function replaceOptions({ el, variables, value }) {
  if (!variables || variables.length <= 1) {
    el.disabled = true;
    return;
  }

  const options = variables.map((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.innerText = name;

    return option;
  });
  el.replaceChildren(...options);

  el.value = value;
  el.disabled = false;
}

export default {
  initialize({ model, signal }) {
    const variables = model.get("variables");

    if (!variables) {
      model.set("value", variables[0]);
      model.save_changes();
    }
  },
  render({ model, el, signal, host }) {
    let variables = model.get("variables");

    const drop_down = document.createElement("select");
    drop_down.classList.add("xdggs-variable-select");
    drop_down.name = "variable";

    let value = model.get("value");
    if (!value || !(value in variables)) {
      value = variables[0];
      model.set("value", value);
      model.save_changes();
    }

    replaceOptions({ el: drop_down, variables, value });

    drop_down.onchange = () => {
      model.set("value", drop_down.value);
      model.save_changes();
    };

    model.on("change:value", () => {
      drop_down.value = model.get("value");
    });
    model.on("change:variables", () => {
      const variables = model.get("variables");
      let value = model.get("value");
      if (!value || !(value in variables)) {
        value = variables[0];
        model.set("value", value);
        model.save_changes();
      }

      replaceOptions({ el: drop_down, variables, value });
    });

    el.appendChild(drop_down);
  },
};
