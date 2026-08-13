export default {
  initialize({ model, signal }) {},
  async render({ model, el, signal, host }) {
    const values = model.get("dimensions");

    el.style.setProperty("display", "grid");
    el.style.setProperty("padding", "5px");
    el.style.setProperty("column-gap", "10px");

    Object.entries(values).forEach(([name, value]) => {
      const label = document.createElement("span");
      label.innerText = name;

      let input = document.createElement("input");
      input.setAttribute("type", "range");
      input.setAttribute("min", 0);
      input.setAttribute("value", 0);
      input.setAttribute("max", value);

      let valueLabel = document.createElement("span");
      valueLabel.innerText = value;

      input.oninput = () => {
        valueLabel.innerText = input.value;
      };

      el.appendChild(label);
      el.appendChild(input);
      el.appendChild(valueLabel);
    });

    el.style.setProperty(
      "grid-template-columns",
      "max-content max-content max-content",
    );
  },
};
