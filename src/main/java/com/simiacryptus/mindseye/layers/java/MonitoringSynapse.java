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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.util.MonitoredItem;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.data.PercentileStatistics;
import com.simiacryptus.util.data.ScalarStatistics;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public final class MonitoringSynapse extends LayerBase implements MonitoredItem {

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
  public Map<CharSequence, Object> getMetrics() {
    @Nonnull final HashMap<CharSequence, Object> map = new HashMap<>();
    map.put("totalBatches", totalBatches);
    map.put("totalItems", totalItems);
    map.put("forward", forwardStatistics.getMetrics());
    map.put("backprop", backpropStatistics.getMetrics());
    return map;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static MonitoringSynapse fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    @Nonnull final MonitoringSynapse obj = new MonitoringSynapse(json);
    obj.totalBatches = json.get("totalBatches").getAsInt();
    obj.totalItems = json.get("totalItems").getAsInt();
    obj.backpropStatistics.readJson(json.getAsJsonObject("backpropStatistics"));
    obj.forwardStatistics.readJson(json.getAsJsonObject("forwardStatistics"));
    return obj;
  }

  @Nonnull
  public MonitoringSynapse addTo(@Nonnull final MonitoredObject obj) {
    addTo(obj, getName());
    return this.addRef();
  }

  public void addTo(@Nonnull MonitoredObject obj, String name) {
    setName(name);
    obj.addObj(getName(), this.addRef());
    RefUtil.freeRef(obj);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert 1 == inObj.length;
    final Result input = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    final TensorList inputdata = input.getData();
    totalBatches++;
    totalItems += inputdata.length();
    forwardStatistics.clear();
    inputdata.stream().parallel().forEach(t -> {
      forwardStatistics.add(t.getData());
      t.freeRef();
    });
    boolean alive = input.isAlive();
    Result.Accumulator accumulator = new Accumulator(input.getAccumulator());
    input.freeRef();
    return new Result(inputdata, accumulator, alive);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("totalBatches", totalBatches);
    json.addProperty("totalItems", totalItems);
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  MonitoringSynapse addRef() {
    return (MonitoringSynapse) super.addRef();
  }

  private class Accumulator extends Result.Accumulator {

    private Result.Accumulator accumulator;

    public Accumulator(Result.Accumulator accumulator) {
      this.accumulator = accumulator;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nullable TensorList data) {
      backpropStatistics.clear();
      TensorList delta = data == null ? null : data.addRef();
      try {
        this.accumulator.accept(buffer, delta);
      } finally {
        this.accumulator.freeRef();
      }
      assert data != null;
      data.stream().parallel().forEach(t -> {
        backpropStatistics.add(t.getData());
        t.freeRef();
      });
      data.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator.freeRef();
    }
  }
}
