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
class ImgConcatLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ImgConcatLayer.class);
  private int maxBands;

  public ImgConcatLayer() {
    setMaxBands(0);
  }

  protected ImgConcatLayer(@Nonnull final JsonObject json) {
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
  public static ImgConcatLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgConcatLayer(json);
  }

  public static @SuppressWarnings("unused")
  ImgConcatLayer[] addRefs(ImgConcatLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgConcatLayer::addRef)
        .toArray((x) -> new ImgConcatLayer[x]);
  }

  public static @SuppressWarnings("unused")
  ImgConcatLayer[][] addRefs(ImgConcatLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgConcatLayer::addRefs)
        .toArray((x) -> new ImgConcatLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert RefArrays.stream(Result.addRefs(inObj)).allMatch(x -> {
      TensorList temp_18_0011 = x.getData();
      boolean temp_18_0004 = temp_18_0011.getDimensions().length == 3;
      if (null != temp_18_0011)
        temp_18_0011.freeRef();
      if (null != x)
        x.freeRef();
      return temp_18_0004;
    }) : "This component is for use mapCoords 3d png tensors only";
    TensorList temp_18_0012 = inObj[0].getData();
    final int numBatches = temp_18_0012.length();
    if (null != temp_18_0012)
      temp_18_0012.freeRef();
    assert RefArrays.stream(Result.addRefs(inObj)).allMatch(x -> {
      TensorList temp_18_0013 = x.getData();
      boolean temp_18_0005 = temp_18_0013.length() == numBatches;
      if (null != temp_18_0013)
        temp_18_0013.freeRef();
      if (null != x)
        x.freeRef();
      return temp_18_0005;
    }) : "All inputs must use same batch size";
    TensorList temp_18_0014 = inObj[0].getData();
    @Nonnull final int[] outputDims = RefArrays.copyOf(temp_18_0014.getDimensions(), 3);
    if (null != temp_18_0014)
      temp_18_0014.freeRef();
    outputDims[2] = RefArrays.stream(Result.addRefs(inObj)).mapToInt(x -> {
      TensorList temp_18_0015 = x.getData();
      int temp_18_0006 = temp_18_0015.getDimensions()[2];
      if (null != temp_18_0015)
        temp_18_0015.freeRef();
      if (null != x)
        x.freeRef();
      return temp_18_0006;
    }).sum();
    if (maxBands > 0)
      outputDims[2] = Math.min(maxBands, outputDims[2]);
    assert RefArrays.stream(Result.addRefs(inObj)).allMatch(x -> {
      TensorList temp_18_0016 = x.getData();
      boolean temp_18_0007 = temp_18_0016.getDimensions()[0] == outputDims[0];
      if (null != temp_18_0016)
        temp_18_0016.freeRef();
      if (null != x)
        x.freeRef();
      return temp_18_0007;
    }) : "Inputs must be same size";
    assert RefArrays.stream(Result.addRefs(inObj)).allMatch(x -> {
      TensorList temp_18_0017 = x.getData();
      boolean temp_18_0008 = temp_18_0017.getDimensions()[1] == outputDims[1];
      if (null != temp_18_0017)
        temp_18_0017.freeRef();
      if (null != x)
        x.freeRef();
      return temp_18_0008;
    }) : "Inputs must be same size";

    @Nonnull final RefList<Tensor> outputTensors = new RefArrayList<>();
    for (int b = 0; b < numBatches; b++) {
      @Nonnull final Tensor outputTensor = new Tensor(outputDims);
      int pos = 0;
      @Nullable final double[] outputTensorData = outputTensor.getData();
      for (int i = 0; i < inObj.length; i++) {
        TensorList temp_18_0018 = inObj[i].getData();
        @Nullable
        Tensor tensor = temp_18_0018.get(b);
        if (null != temp_18_0018)
          temp_18_0018.freeRef();
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
                TensorList temp_18_0019 = inObj[i].getData();
                @Nonnull final Tensor dest = new Tensor(temp_18_0019.getDimensions());
                if (null != temp_18_0019)
                  temp_18_0019.freeRef();
                @Nullable
                double[] tensorData = tensor.getData();
                com.simiacryptus.ref.wrappers.RefSystem.arraycopy(tensorData, pos, dest.getData(), 0, Math.min(dest.length(), tensorData.length - pos));
                pos += dest.length();
                {
                  Tensor temp_18_0001 = dest == null ? null : dest.addRef();
                  if (null != outputTensors2[i])
                    outputTensors2[i].freeRef();
                  outputTensors2[i] = temp_18_0001 == null ? null : temp_18_0001.addRef();
                  if (null != temp_18_0001)
                    temp_18_0001.freeRef();
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
                Tensor[] temp_18_0002 = new Tensor[numBatches];
                if (null != splitData[i])
                  ReferenceCounting.freeRefs(splitData[i]);
                splitData[i] = Tensor.addRefs(temp_18_0002);
                if (null != temp_18_0002)
                  ReferenceCounting.freeRefs(temp_18_0002);
              }
            }
            for (int i = 0; i < inObj.length; i++) {
              for (int b = 0; b < numBatches; b++) {
                {
                  Tensor temp_18_0003 = splitBatches.get(b)[i].addRef();
                  if (null != splitData[i][b])
                    splitData[i][b].freeRef();
                  splitData[i][b] = temp_18_0003 == null ? null : temp_18_0003.addRef();
                  if (null != temp_18_0003)
                    temp_18_0003.freeRef();
                }
              }
            }

            splitBatches.freeRef();
            for (int i = 0; i < inObj.length; i++) {
              @Nonnull
              TensorArray tensorArray = new TensorArray(Tensor.addRefs(splitData[i]));
              inObj[i].accumulate(buffer == null ? null : buffer.addRef(), tensorArray == null ? null : tensorArray);
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
  ImgConcatLayer addRef() {
    return (ImgConcatLayer) super.addRef();
  }
}
