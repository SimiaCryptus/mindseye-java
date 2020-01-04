/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.util.MonitoredItem;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.data.PercentileStatistics;
import com.simiacryptus.util.data.ScalarStatistics;

import javax.annotation.Nonnull;
import java.util.*;

@SuppressWarnings("serial")
public final @com.simiacryptus.ref.lang.RefAware class MonitoringSynapse extends LayerBase implements MonitoredItem {

  private final ScalarStatistics backpropStatistics = new PercentileStatistics();
  private final ScalarStatistics forwardStatistics = new PercentileStatistics();
  private int totalBatches = 0;
  private int totalItems = 0;

  public MonitoringSynapse() {
    super();
  }

  protected MonitoringSynapse(@Nonnull final JsonObject id) {
    super(id);
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefMap<CharSequence, Object> getMetrics() {
    @Nonnull
    final com.simiacryptus.ref.wrappers.RefHashMap<CharSequence, Object> map = new com.simiacryptus.ref.wrappers.RefHashMap<>();
    map.put("totalBatches", totalBatches);
    map.put("totalItems", totalItems);
    map.put("forward", forwardStatistics.getMetrics());
    map.put("backprop", backpropStatistics.getMetrics());
    return map;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static MonitoringSynapse fromJson(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    @Nonnull
    final MonitoringSynapse obj = new MonitoringSynapse(json);
    obj.totalBatches = json.get("totalBatches").getAsInt();
    obj.totalItems = json.get("totalItems").getAsInt();
    obj.backpropStatistics.readJson(json.getAsJsonObject("backpropStatistics"));
    obj.forwardStatistics.readJson(json.getAsJsonObject("forwardStatistics"));
    return obj;
  }

  @Nonnull
  public MonitoringSynapse addTo(@Nonnull final MonitoredObject obj) {
    return addTo(obj, getName());
  }

  @Nonnull
  public MonitoringSynapse addTo(@Nonnull final MonitoredObject obj, final String name) {
    setName(name);
    obj.addObj(getName(), this);
    return this;
  }

  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert 1 == inObj.length;
    final Result input = inObj[0];
    final TensorList inputdata = input.getData();
    System.nanoTime();
    System.nanoTime();
    totalBatches++;
    totalItems += inputdata.length();
    forwardStatistics.clear();
    inputdata.stream().parallel().forEach(t -> {
      forwardStatistics.add(t.getData());
    });
    return new Result(inputdata, (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList data) -> {
      backpropStatistics.clear();
      input.accumulate(buffer, data);
      data.stream().parallel().forEach(t -> {
        backpropStatistics.add(t.getData());
      });
    }) {

      @Override
      public boolean isAlive() {
        return input.isAlive();
      }

      public void _free() {
      }
    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
      DataSerializer dataSerializer) {
    @Nonnull
    final JsonObject json = super.getJsonStub();
    json.addProperty("totalBatches", totalBatches);
    json.addProperty("totalItems", totalItems);
    return json;
  }

  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList();
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") MonitoringSynapse addRef() {
    return (MonitoringSynapse) super.addRef();
  }

  public static @SuppressWarnings("unused") MonitoringSynapse[] addRefs(MonitoringSynapse[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(MonitoringSynapse::addRef)
        .toArray((x) -> new MonitoringSynapse[x]);
  }

  public static @SuppressWarnings("unused") MonitoringSynapse[][] addRefs(MonitoringSynapse[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(MonitoringSynapse::addRefs)
        .toArray((x) -> new MonitoringSynapse[x][]);
  }
}
