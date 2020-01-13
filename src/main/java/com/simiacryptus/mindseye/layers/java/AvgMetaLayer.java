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
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.function.ToDoubleFunction;

@SuppressWarnings("serial")
public class AvgMetaLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(AvgMetaLayer.class);
  @Nullable
  public Tensor lastResult;
  private int minBatchCount = 1;

  public AvgMetaLayer() {
  }

  protected AvgMetaLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> resources) {
    super(json);
    Tensor temp_21_0001 = Tensor.fromJson(json.get("lastResult"), resources);
    if (null != lastResult)
      lastResult.freeRef();
    lastResult = temp_21_0001 == null ? null : temp_21_0001.addRef();
    if (null != temp_21_0001)
      temp_21_0001.freeRef();
    minBatchCount = json.get("minBatchCount").getAsInt();
  }

  public int getMinBatchCount() {
    return minBatchCount;
  }

  @Nonnull
  public AvgMetaLayer setMinBatchCount(final int minBatchCount) {
    this.minBatchCount = minBatchCount;
    return this.addRef();
  }

  @SuppressWarnings("unused")
  public static AvgMetaLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new AvgMetaLayer(json, rs);
  }

  public static @SuppressWarnings("unused") AvgMetaLayer[] addRefs(AvgMetaLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(AvgMetaLayer::addRef).toArray((x) -> new AvgMetaLayer[x]);
  }

  public static @SuppressWarnings("unused") AvgMetaLayer[][] addRefs(AvgMetaLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(AvgMetaLayer::addRefs)
        .toArray((x) -> new AvgMetaLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(final Result... inObj) {
    final Result input = inObj[0].addRef();
    if (null != inObj)
      ReferenceCounting.freeRefs(inObj);
    TensorList inputData = input.getData();
    final int itemCnt = inputData.length();
    @Nullable
    Tensor thisResult;
    boolean passback;
    if (null == lastResult || inputData.length() > minBatchCount) {
      @Nonnull
      final ToDoubleFunction<Coordinate> f = RefUtil
          .wrapInterface((c) -> RefIntStream.range(0, itemCnt).mapToDouble(RefUtil.wrapInterface(dataIndex -> {
            Tensor tensor = inputData.get(dataIndex);
            double temp_21_0005 = tensor.get(c);
            if (null != tensor)
              tensor.freeRef();
            return temp_21_0005;
          }, inputData == null ? null : inputData.addRef())).sum() / itemCnt,
              inputData == null ? null : inputData.addRef());
      Tensor tensor = inputData.get(0);
      thisResult = tensor.mapCoords(f);
      if (null != tensor)
        tensor.freeRef();
      passback = true;
      Tensor temp_21_0002 = thisResult == null ? null : thisResult.addRef();
      if (null != lastResult)
        lastResult.freeRef();
      lastResult = temp_21_0002 == null ? null : temp_21_0002.addRef();
      if (null != temp_21_0002)
        temp_21_0002.freeRef();
    } else {
      passback = false;
      thisResult = lastResult == null ? null : lastResult.addRef();
    }
    if (null != inputData)
      inputData.freeRef();
    try {
      try {
        return new Result(new TensorArray(thisResult == null ? null : thisResult.addRef()), new Result.Accumulator() {
          {
          }

          @Override
          public void accept(DeltaSet<UUID> buffer, TensorList data) {
            if (passback && input.isAlive()) {
              @Nullable
              final Tensor delta = data.get(0);
              @Nonnull
              final Tensor feedback[] = new Tensor[itemCnt];
              RefArrays.parallelSetAll(Tensor.addRefs(feedback),
                  RefUtil.wrapInterface(i -> new Tensor(delta.getDimensions()), delta == null ? null : delta.addRef()));
              thisResult.coordStream(true)
                  .forEach(RefUtil.wrapInterface((Consumer<? super Coordinate>) (inputCoord) -> {
                    for (int inputItem = 0; inputItem < itemCnt; inputItem++) {
                      feedback[inputItem].add(inputCoord, delta.get(inputCoord) / itemCnt);
                    }
                  }, delta == null ? null : delta.addRef(), Tensor.addRefs(feedback)));
              if (null != delta)
                delta.freeRef();
              @Nonnull
              TensorArray tensorArray = new TensorArray(Tensor.addRefs(feedback));
              ReferenceCounting.freeRefs(feedback);
              input.accumulate(buffer == null ? null : buffer.addRef(), tensorArray == null ? null : tensorArray);
            }
            if (null != data)
              data.freeRef();
            if (null != buffer)
              buffer.freeRef();
          }

          public @SuppressWarnings("unused") void _free() {
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
        if (null != thisResult)
          thisResult.freeRef();
      }
    } finally {
      if (null != input)
        input.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull
    final JsonObject json = super.getJsonStub();
    if (null != lastResult) {
      json.add("lastResult", lastResult.getJson(resources, dataSerializer));
    }
    json.addProperty("minBatchCount", minBatchCount);
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public void _free() {
    if (null != lastResult)
      lastResult.freeRef();
    lastResult = null;
    super._free();
  }

  public @Override @SuppressWarnings("unused") AvgMetaLayer addRef() {
    return (AvgMetaLayer) super.addRef();
  }
}
