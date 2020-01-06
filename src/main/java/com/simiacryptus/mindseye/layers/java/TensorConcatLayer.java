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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrayList;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public @RefAware
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
  public static TensorConcatLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new TensorConcatLayer(json);
  }

  public static @SuppressWarnings("unused")
  TensorConcatLayer[] addRefs(TensorConcatLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(TensorConcatLayer::addRef)
        .toArray((x) -> new TensorConcatLayer[x]);
  }

  public static @SuppressWarnings("unused")
  TensorConcatLayer[][] addRefs(TensorConcatLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(TensorConcatLayer::addRefs)
        .toArray((x) -> new TensorConcatLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    TensorList temp_09_0009 = inObj[0].getData();
    final int numBatches = temp_09_0009.length();
    if (null != temp_09_0009)
      temp_09_0009.freeRef();
    assert RefArrays.stream(Result.addRefs(inObj)).allMatch(x -> {
      TensorList temp_09_0010 = x.getData();
      boolean temp_09_0004 = temp_09_0010.length() == numBatches;
      if (null != temp_09_0010)
        temp_09_0010.freeRef();
      if (null != x)
        x.freeRef();
      return temp_09_0004;
    }) : "All inputs must use same batch size";
    int[] outputDims = new int[]{
        RefArrays.stream(Result.addRefs(inObj)).mapToInt(x -> {
          TensorList temp_09_0011 = x.getData();
          int temp_09_0005 = Tensor.length(temp_09_0011.getDimensions());
          if (null != temp_09_0011)
            temp_09_0011.freeRef();
          if (null != x)
            x.freeRef();
          return temp_09_0005;
        }).sum()};

    @Nonnull final RefList<Tensor> outputTensors = new RefArrayList<>();
    for (int b = 0; b < numBatches; b++) {
      @Nonnull final Tensor outputTensor = new Tensor(outputDims);
      int pos = 0;
      @Nullable final double[] outputTensorData = outputTensor.getData();
      for (int i = 0; i < inObj.length; i++) {
        TensorList temp_09_0012 = inObj[i].getData();
        @Nullable
        Tensor tensor = temp_09_0012.get(b);
        if (null != temp_09_0012)
          temp_09_0012.freeRef();
        @Nullable final double[] data = tensor.getData();
        if (null != tensor)
          tensor.freeRef();
        com.simiacryptus.ref.wrappers.RefSystem.arraycopy(data, 0, outputTensorData, pos, Math.min(data.length, outputTensorData.length - pos));
        pos += data.length;
      }
      outputTensors.add(outputTensor == null ? null : outputTensor);
    }
    try {
      try {
        return new Result(new TensorArray(outputTensors.toArray(new Tensor[]{})), new Result.Accumulator() {
          {
            Result.addRefs(inObj);
          }

          @Override
          public void accept(DeltaSet<UUID> buffer, TensorList data) {
            assert numBatches == data.length();

            @Nonnull final RefList<Tensor[]> splitBatches = new RefArrayList<>();
            for (int b = 0; b < numBatches; b++) {
              @Nullable final Tensor tensor = data.get(b);
              @Nonnull final Tensor[] outputTensors2 = new Tensor[inObj.length];
              int pos = 0;
              for (int i = 0; i < inObj.length; i++) {
                TensorList temp_09_0013 = inObj[i].getData();
                @Nonnull final Tensor dest = new Tensor(temp_09_0013.getDimensions());
                if (null != temp_09_0013)
                  temp_09_0013.freeRef();
                @Nullable
                double[] tensorData = tensor.getData();
                com.simiacryptus.ref.wrappers.RefSystem.arraycopy(tensorData, pos, dest.getData(), 0, Math.min(dest.length(), tensorData.length - pos));
                pos += dest.length();
                {
                  Tensor temp_09_0001 = dest == null ? null : dest.addRef();
                  if (null != outputTensors2[i])
                    outputTensors2[i].freeRef();
                  outputTensors2[i] = temp_09_0001 == null ? null : temp_09_0001.addRef();
                  if (null != temp_09_0001)
                    temp_09_0001.freeRef();
                }
                dest.freeRef();
              }
              if (null != tensor)
                tensor.freeRef();
              splitBatches.add(Tensor.addRefs(outputTensors2));
              ReferenceCounting.freeRefs(outputTensors2);
            }

            if (null != data)
              data.freeRef();
            @Nonnull final Tensor[][] splitData = new Tensor[inObj.length][];
            for (int i = 0; i < splitData.length; i++) {
              {
                Tensor[] temp_09_0002 = new Tensor[numBatches];
                if (null != splitData[i])
                  ReferenceCounting.freeRefs(splitData[i]);
                splitData[i] = Tensor.addRefs(temp_09_0002);
                if (null != temp_09_0002)
                  ReferenceCounting.freeRefs(temp_09_0002);
              }
            }
            for (int i = 0; i < inObj.length; i++) {
              for (int b = 0; b < numBatches; b++) {
                {
                  Tensor temp_09_0003 = splitBatches.get(b)[i].addRef();
                  if (null != splitData[i][b])
                    splitData[i][b].freeRef();
                  splitData[i][b] = temp_09_0003 == null ? null : temp_09_0003.addRef();
                  if (null != temp_09_0003)
                    temp_09_0003.freeRef();
                }
              }
            }

            splitBatches.freeRef();
            for (int i = 0; i < inObj.length; i++) {
              TensorArray wrap = new TensorArray(Tensor.addRefs(splitData[i]));
              inObj[i].accumulate(buffer == null ? null : buffer.addRef(), wrap == null ? null : wrap.addRef());
              if (0 < wrap.currentRefCount()) {
                ReferenceCounting.freeRefs(splitData);
                RuntimeException temp_09_0007 = new RuntimeException(
                    inObj[i].getClass() + " leak: " + wrap.currentRefCount());
                if (null != wrap)
                  wrap.freeRef();
                if (null != buffer)
                  buffer.freeRef();
                throw temp_09_0007;
              }
              if (null != wrap)
                wrap.freeRef();
            }
            if (null != buffer)
              buffer.freeRef();
            ReferenceCounting.freeRefs(splitData);
          }

          public @SuppressWarnings("unused")
          void _free() {
            ReferenceCounting.freeRefs(inObj);
          }
        }) {

          {
            Result.addRefs(inObj);
          }

          @Override
          public boolean isAlive() {
            for (@Nonnull final Result element : inObj)
              if (element.isAlive()) {
                return true;
              }
            return false;
          }

          public void _free() {
            ReferenceCounting.freeRefs(inObj);
          }

        };
      } finally {
        ReferenceCounting.freeRefs(inObj);
      }
    } finally {
      outputTensors.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull
    JsonObject json = super.getJsonStub();
    json.addProperty("maxBands", maxBands);
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
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
