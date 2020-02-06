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
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.function.ToDoubleFunction;

@SuppressWarnings("serial")
public class SumMetaLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SumMetaLayer.class);
  @Nullable
  private Tensor lastResult;
  private int minBatches = 1;

  public SumMetaLayer() {
  }

  protected SumMetaLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> resources) {
    super(json);
    Tensor temp_13_0001 = Tensor.fromJson(json.get("lastResult"), resources);
    if (null != lastResult)
      lastResult.freeRef();
    lastResult = temp_13_0001 == null ? null : temp_13_0001.addRef();
    if (null != temp_13_0001)
      temp_13_0001.freeRef();
    minBatches = json.get("minBatches").getAsInt();
  }

  public int getMinBatches() {
    return minBatches;
  }

  public void setMinBatches(int minBatches) {
    this.minBatches = minBatches;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static SumMetaLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SumMetaLayer(json, rs);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (1 != inObj.length) {
      RefUtil.freeRef(inObj);
      throw new IllegalArgumentException();
    }
    final Result input = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    TensorList inputData = input.getData();
    final int itemCnt = inputData.length();
    if (null == lastResult || minBatches < itemCnt) {
      @Nonnull final ToDoubleFunction<Coordinate> f = RefUtil
          .wrapInterface(c -> RefIntStream.range(0, itemCnt).mapToDouble(RefUtil.wrapInterface(dataIndex -> {
            Tensor tensor = inputData.get(dataIndex);
            double temp_13_0004 = tensor.get(c);
            tensor.freeRef();
            return temp_13_0004;
          }, inputData.addRef())).sum(), inputData.addRef());
      Tensor temp_13_0005 = inputData.get(0);
      Tensor temp_13_0002 = temp_13_0005.mapCoords(f);
      temp_13_0005.freeRef();
      if (null != lastResult)
        lastResult.freeRef();
      lastResult = temp_13_0002.addRef();
      temp_13_0002.freeRef();
    }
    inputData.freeRef();
    try {
      Result.Accumulator accumulator = new Result.Accumulator() {
        {
          input.addRef();
        }

        @Override
        public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList data) {
          if (input.isAlive()) {
            @Nullable final Tensor delta = data.get(0);
            @Nonnull final Tensor feedback[] = new Tensor[itemCnt];
            RefArrays.parallelSetAll(RefUtil.addRefs(feedback),
                RefUtil.wrapInterface(i -> new Tensor(delta.getDimensions()), delta.addRef()));
            delta.coordStream(false).forEach(RefUtil.wrapInterface((Consumer<? super Coordinate>) inputCoord -> {
              for (int inputItem = 0; inputItem < itemCnt; inputItem++) {
                feedback[inputItem].add(inputCoord, delta.get(inputCoord));
              }
            }, RefUtil.addRefs(feedback), delta.addRef()));
            delta.freeRef();
            @Nonnull
            TensorArray tensorArray = new TensorArray(RefUtil.addRefs(feedback));
            RefUtil.freeRef(feedback);
            input.accumulate(buffer == null ? null : buffer.addRef(), tensorArray);
          }
          data.freeRef();
          if (null != buffer)
            buffer.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
          super._free();
          input.freeRef();
        }
      };
      return new Result(new TensorArray(lastResult == null ? null : lastResult.addRef()), accumulator) {
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
      input.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    if (null != lastResult) {
      json.add("lastResult", lastResult.getJson(resources, dataSerializer));
    }
    json.addProperty("minBatches", minBatches);
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

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  SumMetaLayer addRef() {
    return (SumMetaLayer) super.addRef();
  }
}
