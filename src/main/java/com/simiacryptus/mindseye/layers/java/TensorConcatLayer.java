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

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.UUID;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware
class TensorConcatLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(TensorConcatLayer.class);
  private int maxBands;

  public TensorConcatLayer() {
    setMaxBands(0);
  }

  protected TensorConcatLayer(@Nonnull final JsonObject json) {
    super(json);
    JsonElement maxBands = json.get("maxBands");
    if (null != maxBands)
      setMaxBands(maxBands.getAsInt());
  }

  public int getMaxBands() {
    return maxBands;
  }

  @Nonnull
  public void setMaxBands(int maxBands) {
    this.maxBands = maxBands;
  }

  @SuppressWarnings("unused")
  public static TensorConcatLayer fromJson(@Nonnull final JsonObject json,
                                           com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new TensorConcatLayer(json);
  }

  public static @SuppressWarnings("unused")
  TensorConcatLayer[] addRefs(TensorConcatLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(TensorConcatLayer::addRef)
        .toArray((x) -> new TensorConcatLayer[x]);
  }

  public static @SuppressWarnings("unused")
  TensorConcatLayer[][] addRefs(TensorConcatLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(TensorConcatLayer::addRefs)
        .toArray((x) -> new TensorConcatLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final int numBatches = inObj[0].getData().length();
    assert com.simiacryptus.ref.wrappers.RefArrays.stream(inObj)
        .allMatch(x -> x.getData().length() == numBatches) : "All inputs must use same batch size";
    int[] outputDims = new int[]{com.simiacryptus.ref.wrappers.RefArrays.stream(inObj)
        .mapToInt(x -> Tensor.length(x.getData().getDimensions())).sum()};

    @Nonnull final com.simiacryptus.ref.wrappers.RefList<Tensor> outputTensors = new com.simiacryptus.ref.wrappers.RefArrayList<>();
    for (int b = 0; b < numBatches; b++) {
      @Nonnull final Tensor outputTensor = new Tensor(outputDims);
      int pos = 0;
      @Nullable final double[] outputTensorData = outputTensor.getData();
      for (int i = 0; i < inObj.length; i++) {
        @Nullable
        Tensor tensor = inObj[i].getData().get(b);
        @Nullable final double[] data = tensor.getData();
        System.arraycopy(data, 0, outputTensorData, pos, Math.min(data.length, outputTensorData.length - pos));
        pos += data.length;
      }
      outputTensors.add(outputTensor);
    }
    return new Result(new TensorArray(outputTensors.toArray(new Tensor[]{})),
        (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList data) -> {
          assert numBatches == data.length();

          @Nonnull final com.simiacryptus.ref.wrappers.RefList<Tensor[]> splitBatches = new com.simiacryptus.ref.wrappers.RefArrayList<>();
          for (int b = 0; b < numBatches; b++) {
            @Nullable final Tensor tensor = data.get(b);
            @Nonnull final Tensor[] outputTensors2 = new Tensor[inObj.length];
            int pos = 0;
            for (int i = 0; i < inObj.length; i++) {
              @Nonnull final Tensor dest = new Tensor(inObj[i].getData().getDimensions());
              @Nullable
              double[] tensorData = tensor.getData();
              System.arraycopy(tensorData, pos, dest.getData(), 0, Math.min(dest.length(), tensorData.length - pos));
              pos += dest.length();
              outputTensors2[i] = dest;
            }
            splitBatches.add(outputTensors2);
          }

          @Nonnull final Tensor[][] splitData = new Tensor[inObj.length][];
          for (int i = 0; i < splitData.length; i++) {
            splitData[i] = new Tensor[numBatches];
          }
          for (int i = 0; i < inObj.length; i++) {
            for (int b = 0; b < numBatches; b++) {
              splitData[i][b] = splitBatches.get(b)[i];
            }
          }

          for (int i = 0; i < inObj.length; i++) {
            TensorArray wrap = new TensorArray(splitData[i]);
            inObj[i].accumulate(buffer, wrap);
            if (0 < wrap.currentRefCount()) {
              throw new RuntimeException(inObj[i].getClass() + " leak: " + wrap.currentRefCount());
            }
          }
        }) {

      @Override
      public boolean isAlive() {
        for (@Nonnull final Result element : inObj)
          if (element.isAlive()) {
            return true;
          }
        return false;
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
    JsonObject json = super.getJsonStub();
    json.addProperty("maxBands", maxBands);
    return json;
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  TensorConcatLayer addRef() {
    return (TensorConcatLayer) super.addRef();
  }
}
