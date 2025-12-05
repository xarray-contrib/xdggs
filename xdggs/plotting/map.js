async function unpack_models(model_ids, manager) {
  return Promise.all(
    model_ids.map((id) => manager.get_model(id.slice("IPY_MODEL_".length))),
  );
}

async function create_child_views(models, manager) {
  return Promise.all(models.map((m) => manager.create_view(m)));
}

function construct_slider_row(name, slider_view) {
  let row = document.createElement("div");
  row.style.setProperty("display", "grid");
  row.style.setProperty("padding", "3px");
  row.style.setProperty("grid-template-columns", "50px 1fr 5fr 1fr");

  // play button for animation
  /* let button = document.createElement("button");
   * button.innerHTML = "▶";
   * button.style.setProperty("width", "40px");
   * row.appendChild(button);
   */

  // dimension name
  let name_el = document.createElement("div");
  name_el.innerHTML = name;
  name_el.style.setProperty("align-content", "center");
  name_el.style.setProperty("padding-left", "2px");
  row.appendChild(name_el);

  // slider widget
  row.appendChild(slider_view.el);

  // value label
  let readout = document.createElement("div");
  readout.style.setProperty("padding-left", "2px");
  readout.style.setProperty("align-content", "center");
  readout.id = `${name}_label`;
  readout.innerHTML = "value";
  row.appendChild(readout);

  return row;
}

async function create_controls(model) {
  const controls = document.createElement("div");
  controls.style.setProperty("padding", "5px");
  controls.style.setProperty("display", "grid");
  controls.style.setProperty("grid-template-columns", "2fr 3fr 3fr");

  // variable drop down
  const variables = model.get("variables");
  let variable_model = await unpack_models([variables], model.widget_manager);
  let variable_views = await create_child_views(
    variable_model,
    model.widget_manager,
  );
  let variable_view = variable_views[0];
  controls.appendChild(variable_view.el);

  // sliders
  let dimension_sliders = document.createElement("div");
  dimension_sliders.style.setProperty("display", "grid");
  dimension_sliders.style.setProperty("grid-template-columns", "1fr");

  const slider_ids = model.get("sliders");
  const slider_models = await unpack_models(
    Object.values(slider_ids),
    model.widget_manager,
  );
  const slider_views = await create_child_views(
    slider_models,
    model.widget_manager,
  );

  const dimension_indices = model.get("dimensions");
  const coordinate_values = model.get("coordinates");

  let zipped = Object.keys(dimension_indices).map((name, index) => [
    name,
    slider_views[index],
    slider_models[index],
    dimension_indices[name],
    coordinate_values[name],
  ]);
  zipped.forEach(
    ([name, slider_view, slider_model, dimension_index, values]) => {
      let row = construct_slider_row(name, slider_view);

      let label = row.children[2];
      label.innerHTML = values[0];

      // link slider change with label (index => values[index] => label.value)
      slider_model.on("change:value", (model, new_value, context) => {
        label.innerHTML = values[new_value];
      });

      dimension_sliders.appendChild(row);
    },
  );

  controls.appendChild(dimension_sliders);

  // colorbar
  const colorbar = model.get("colorbar");
  let colorbar_models = await unpack_models([colorbar], model.widget_manager);
  let colorbar_views = await create_child_views(
    colorbar_models,
    model.widget_manager,
  );
  let colorbar_view = colorbar_views[0];
  controls.appendChild(colorbar_view.el);

  return controls;
}

async function render({ model, el }) {
  // map
  let map_id = model.get("map");
  let map_models = await unpack_models([map_id], model.widget_manager);
  let map_views = await create_child_views(map_models, model.widget_manager);
  let map_view = map_views[0];
  map_view.el.style.setProperty("height", "80%");
  el.appendChild(map_view.el);

  // controls
  const controls = await create_controls(model);
  el.appendChild(controls);
}

export default { render };
