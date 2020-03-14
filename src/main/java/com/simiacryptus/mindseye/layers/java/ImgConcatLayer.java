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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrayList;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefSystem;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public class ImgConcatLayer extends LayerBase {

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

  public void setMaxBands(int maxBands) {
    this.maxBands = maxBands;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ImgConcatLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgConcatLayer(json);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert RefArrays.stream(RefUtil.addRef(inObj)).allMatch(x -> {
      TensorList temp_18_0011 = x.getData();
      boolean temp_18_0004 = temp_18_0011.getDimensions().length == 3;
      temp_18_0011.freeRef();
      x.freeRef();
      return temp_18_0004;
    }) : "This component is for use mapCoords 3d png tensors only";
    TensorList temp_18_0012 = inObj[0].getData();
    final int numBatches = temp_18_0012.length();
    temp_18_0012.freeRef();
    assert RefArrays.stream(RefUtil.addRef(inObj)).allMatch(x -> {
      TensorList temp_18_0013 = x.getData();
      boolean temp_18_0005 = temp_18_0013.length() == numBatches;
      temp_18_0013.freeRef();
      x.freeRef();
      return temp_18_0005;
    }) : "All inputs must use same batch size";
    TensorList temp_18_0014 = inObj[0].getData();
    @Nonnull final int[] outputDims = RefArrays.copyOf(temp_18_0014.getDimensions(), 3);
    temp_18_0014.freeRef();
    outputDims[2] = RefArrays.stream(RefUtil.addRef(inObj)).mapToInt(x -> {
      TensorList temp_18_0015 = x.getData();
      int temp_18_0006 = temp_18_0015.getDimensions()[2];
      temp_18_0015.freeRef();
      x.freeRef();
      return temp_18_0006;
    }).sum();
    if (maxBands > 0)
      outputDims[2] = Math.min(maxBands, outputDims[2]);
    assert RefArrays.stream(RefUtil.addRef(inObj)).allMatch(x -> {
      TensorList temp_18_0016 = x.getData();
      boolean temp_18_0007 = temp_18_0016.getDimensions()[0] == outputDims[0];
      temp_18_0016.freeRef();
      x.freeRef();
      return temp_18_0007;
    }) : "Inputs must be same size";
    assert RefArrays.stream(RefUtil.addRef(inObj)).allMatch(x -> {
      TensorList temp_18_0017 = x.getData();
      boolean temp_18_0008 = temp_18_0017.getDimensions()[1] == outputDims[1];
      temp_18_0017.freeRef();
      x.freeRef();
      return temp_18_0008;
    }) : "Inputs must be same size";

    TensorArray data = fwd(numBatches, outputDims, RefUtil.addRef(inObj));
    boolean alive = Result.anyAlive(RefUtil.addRef(inObj));
    Accumulator accumulator = new Accumulator(numBatches, inObj);
    return new Result(data, accumulator, alive);
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
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgConcatLayer addRef() {
    return (ImgConcatLayer) super.addRef();
  }

  @NotNull
  private TensorArray fwd(int numBatches, int[] outputDims, @Nonnull Result[] inObj) {
    @Nonnull final RefList<Tensor> outputTensors = new RefArrayList<>();
    for (int b = 0; b < numBatches; b++) {
      @Nonnull final Tensor outputTensor = new Tensor(outputDims);
      int pos = 0;
      @Nullable final double[] outputTensorData = outputTensor.getData();
      for (int i = 0; i < inObj.length; i++) {
        TensorList temp_18_0018 = inObj[i].getData();
        @Nullable
        Tensor tensor = temp_18_0018.get(b);
        temp_18_0018.freeRef();
        @Nullable final double[] data = tensor.getData();
        tensor.freeRef();
        RefSystem.arraycopy(data, 0, outputTensorData, pos,
            Math.min(data.length, outputTensorData.length - pos));
        pos += data.length;
      }
      outputTensors.add(outputTensor);
    }
    RefUtil.freeRef(inObj);
    TensorArray tensorArray = new TensorArray(outputTensors.toArray(new Tensor[]{}));
    outputTensors.freeRef();
    return tensorArray;
  }

  private static class Accumulator extends Result.Accumulator {

    private final int numBatches;
    private final Result[] inObj;

    public Accumulator(int numBatches, Result... inObj) {
      this.numBatches = numBatches;
      this.inObj = inObj;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList data) {
      assert numBatches == data.length();

      @Nonnull final RefList<Tensor[]> splitBatches = new RefArrayList<>();
      for (int b = 0; b < numBatches; b++) {
        @Nullable final Tensor tensor = data.get(b);
        @Nonnull final Tensor[] outputTensors2 = new Tensor[inObj.length];
        int pos = 0;
        for (int i = 0; i < inObj.length; i++) {
          TensorList temp_18_0019 = inObj[i].getData();
          @Nonnull final Tensor dest = new Tensor(temp_18_0019.getDimensions());
          temp_18_0019.freeRef();
          @Nullable
          double[] tensorData = tensor.getData();
          RefSystem.arraycopy(tensorData, pos, dest.getData(), 0,
              Math.min(dest.length(), tensorData.length - pos));
          pos += dest.length();
          RefUtil.set(outputTensors2, i, dest.addRef());
          dest.freeRef();
        }
        tensor.freeRef();
        splitBatches.add(RefUtil.addRef(outputTensors2));
        RefUtil.freeRef(outputTensors2);
      }

      data.freeRef();
      @Nonnull final Tensor[][] splitData = new Tensor[inObj.length][];
      for (int i = 0; i < splitData.length; i++) {
        RefUtil.set(splitData, i, new Tensor[numBatches]);
      }
      for (int i = 0; i < inObj.length; i++) {
        for (int b = 0; b < numBatches; b++) {
          Tensor[] tensors = splitBatches.get(b);
          assert tensors != null;
          RefUtil.set(splitData[i], b, tensors[i].addRef());
          RefUtil.freeRef(tensors);
        }
      }

      splitBatches.freeRef();
      for (int i = 0; i < inObj.length; i++) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(RefUtil.addRef(splitData[i]));
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        Result.Accumulator accumulator = inObj[i].getAccumulator();
        try {
          accumulator.accept(buffer1, tensorArray);
        } finally {
          accumulator.freeRef();
        }
      }
      if (null != buffer)
        buffer.freeRef();
      RefUtil.freeRef(splitData);
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      RefUtil.freeRef(inObj);
    }
  }
}
