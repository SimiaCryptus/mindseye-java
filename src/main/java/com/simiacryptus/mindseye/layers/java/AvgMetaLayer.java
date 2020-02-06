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
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
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

  public void setMinBatchCount(int minBatchCount) {
    this.minBatchCount = minBatchCount;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static AvgMetaLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new AvgMetaLayer(json, rs);
  }

  @Nonnull
  @Override
  public Result eval(@Nullable final Result... inObj) {
    assert inObj != null;
    final Result input = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    TensorList inputData = input.getData();
    final int itemCnt = inputData.length();
    AtomicBoolean passback = new AtomicBoolean(false);
    @Nullable Tensor thisResult = getTensor(inputData, itemCnt, passback);
    try {
      Result.Accumulator accumulator = new Result.Accumulator() {
        {
          input.addRef();
          thisResult.addRef();
        }

        @Override
        public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList data) {
          if (passback.get() && input.isAlive()) {
            @Nullable final Tensor delta = data.get(0);
            @Nonnull final Tensor feedback[] = new Tensor[itemCnt];
            RefArrays.parallelSetAll(RefUtil.addRefs(feedback),
                RefUtil.wrapInterface(i -> new Tensor(delta.getDimensions()), delta.addRef()));
            thisResult.coordStream(true)
                .forEach(RefUtil.wrapInterface(inputCoord -> {
                  for (int inputItem = 0; inputItem < itemCnt; inputItem++) {
                    feedback[inputItem].add(inputCoord, delta.get(inputCoord) / itemCnt);
                  }
                }, delta, RefUtil.addRefs(feedback)));
            @Nonnull
            TensorArray tensorArray = new TensorArray(feedback);
            input.accumulate(buffer == null ? null : buffer.addRef(), tensorArray);
          }
          data.freeRef();
          if (null != buffer)
            buffer.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
          super._free();
          thisResult.freeRef();
          input.freeRef();
        }
      };
      return new Result(new TensorArray(thisResult == null ? null : thisResult.addRef()), accumulator) {
        {
          input.addRef();
        }
        @Override
        public boolean isAlive() {
          return input.isAlive();
        }

        @Override
        public void _free() {
          input.freeRef();
          super._free();
        }
      };
    } finally {
      if (null != thisResult)
        thisResult.freeRef();
      input.freeRef();
    }
  }

  @org.jetbrains.annotations.Nullable
  public Tensor getTensor(TensorList inputData, int itemCnt, AtomicBoolean passback) {
    try {
      if (null == lastResult || inputData.length() > minBatchCount) {
        @Nonnull final ToDoubleFunction<Coordinate> f = RefUtil
            .wrapInterface(c -> RefIntStream.range(0, itemCnt).mapToDouble(RefUtil.wrapInterface(dataIndex -> {
                  Tensor tensor = inputData.get(dataIndex);
                  double temp_21_0005 = tensor.get(c);
                  tensor.freeRef();
                  return temp_21_0005;
                }, inputData.addRef())).sum() / itemCnt,
                inputData.addRef());
        passback.set(true);
        clearLastResult();
        Tensor tensor = inputData.get(0);
        Tensor thisResult = tensor.mapCoords(f);
        tensor.freeRef();
        return thisResult;
      } else {
        passback.set(false);
        return lastResult == null ? null : lastResult.addRef();
      }
    } finally {
      inputData.freeRef();
    }
  }

  @RefIgnore
  public void clearLastResult() {
    if (null != lastResult) {
      lastResult.freeRef();
      lastResult = null;
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
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
    if (null != lastResult) {
      lastResult.freeRef();
      lastResult = null;
    }
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  AvgMetaLayer addRef() {
    return (AvgMetaLayer) super.addRef();
  }
}
