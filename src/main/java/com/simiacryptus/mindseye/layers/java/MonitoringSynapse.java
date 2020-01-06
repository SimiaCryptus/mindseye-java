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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefHashMap;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.util.MonitoredItem;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.data.PercentileStatistics;
import com.simiacryptus.util.data.ScalarStatistics;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public final @RefAware
class MonitoringSynapse extends LayerBase implements MonitoredItem {

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

  public static @SuppressWarnings("unused")
  MonitoringSynapse[] addRefs(MonitoringSynapse[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(MonitoringSynapse::addRef)
        .toArray((x) -> new MonitoringSynapse[x]);
  }

  public static @SuppressWarnings("unused")
  MonitoringSynapse[][] addRefs(MonitoringSynapse[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(MonitoringSynapse::addRefs)
        .toArray((x) -> new MonitoringSynapse[x][]);
  }

  @Nonnull
  public MonitoringSynapse addTo(@Nonnull final MonitoredObject obj) {
    MonitoringSynapse temp_37_0003 = addTo(obj == null ? null : obj, getName());
    return temp_37_0003;
  }

  @Nonnull
  public MonitoringSynapse addTo(@Nonnull final MonitoredObject obj, final String name) {
    RefUtil.freeRef(setName(name));
    RefUtil.freeRef(obj.addObj(getName(), this.addRef()));
    obj.freeRef();
    return this.addRef();
  }

  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert 1 == inObj.length;
    final Result input = inObj[0].addRef();
    ReferenceCounting.freeRefs(inObj);
    final TensorList inputdata = input.getData();
    com.simiacryptus.ref.wrappers.RefSystem.nanoTime();
    com.simiacryptus.ref.wrappers.RefSystem.nanoTime();
    totalBatches++;
    totalItems += inputdata.length();
    forwardStatistics.clear();
    inputdata.stream().parallel().forEach(t -> {
      forwardStatistics.add(t.getData());
      if (null != t)
        t.freeRef();
    });
    try {
      try {
        return new Result(inputdata, new Result.Accumulator() {
          {
          }

          @Override
          public void accept(DeltaSet<UUID> buffer, TensorList data) {
            backpropStatistics.clear();
            input.accumulate(buffer == null ? null : buffer.addRef(), data == null ? null : data.addRef());
            if (null != buffer)
              buffer.freeRef();
            data.stream().parallel().forEach(t -> {
              backpropStatistics.add(t.getData());
              if (null != t)
                t.freeRef();
            });
            if (null != data)
              data.freeRef();
          }

          public @SuppressWarnings("unused")
          void _free() {
          }
        }) {

          {
          }

          @Override
          public boolean isAlive() {
            return input.isAlive();
          }

          public void _free() {
          }
        };
      } finally {
        if (null != inputdata)
          inputdata.freeRef();
      }
    } finally {
      if (null != input)
        input.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("totalBatches", totalBatches);
    json.addProperty("totalItems", totalItems);
    return json;
  }

  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  MonitoringSynapse addRef() {
    return (MonitoringSynapse) super.addRef();
  }
}
