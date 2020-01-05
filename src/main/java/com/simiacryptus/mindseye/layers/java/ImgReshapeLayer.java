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
import com.simiacryptus.ref.wrappers.RefArrayList;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public @RefAware
class ImgReshapeLayer extends LayerBase {

  private final boolean expand;
  private final int kernelSizeX;
  private final int kernelSizeY;

  public ImgReshapeLayer(final int kernelSizeX, final int kernelSizeY, final boolean expand) {
    super();
    this.kernelSizeX = kernelSizeX;
    this.kernelSizeY = kernelSizeY;
    this.expand = expand;
  }

  protected ImgReshapeLayer(@Nonnull final JsonObject json) {
    super(json);
    kernelSizeX = json.getAsJsonPrimitive("kernelSizeX").getAsInt();
    kernelSizeY = json.getAsJsonPrimitive("kernelSizeY").getAsInt();
    expand = json.getAsJsonPrimitive("expandPlasma").getAsBoolean();
  }

  @Nonnull
  public static Tensor copyCondense(@Nonnull final Tensor inputData, @Nonnull final Tensor outputData) {
    @Nonnull final int[] inDim = inputData.getDimensions();
    @Nonnull final int[] outDim = outputData.getDimensions();
    assert 3 == inDim.length;
    assert 3 == outDim.length;
    assert inDim[0] >= outDim[0];
    assert inDim[1] >= outDim[1];
    assert inDim[2] < outDim[2];
    assert 0 == inDim[0] % outDim[0];
    assert 0 == inDim[1] % outDim[1];
    final int kernelSizeX = inDim[0] / outDim[0];
    final int kernelSizeY = inDim[0] / outDim[0];
    int index = 0;
    @Nullable final double[] outputDataData = outputData.getData();
    for (int z = 0; z < inDim[2]; z++) {
      for (int xx = 0; xx < kernelSizeX; xx++) {
        for (int yy = 0; yy < kernelSizeY; yy++) {
          for (int y = 0; y < inDim[1]; y += kernelSizeY) {
            for (int x = 0; x < inDim[0]; x += kernelSizeX) {
              outputDataData[index++] = inputData.get(x + xx, y + yy, z);
            }
          }
        }
      }
    }
    inputData.freeRef();
    return outputData;
  }

  @Nonnull
  public static Tensor copyExpand(@Nonnull final Tensor inputData, @Nonnull final Tensor outputData) {
    @Nonnull final int[] inDim = inputData.getDimensions();
    @Nonnull final int[] outDim = outputData.getDimensions();
    assert 3 == inDim.length;
    assert 3 == outDim.length;
    assert inDim[0] <= outDim[0];
    assert inDim[1] <= outDim[1];
    assert inDim[2] > outDim[2];
    assert 0 == outDim[0] % inDim[0];
    assert 0 == outDim[1] % inDim[1];
    final int kernelSizeX = outDim[0] / inDim[0];
    final int kernelSizeY = outDim[0] / inDim[0];
    int index = 0;
    for (int z = 0; z < outDim[2]; z++) {
      for (int xx = 0; xx < kernelSizeX; xx++) {
        for (int yy = 0; yy < kernelSizeY; yy++) {
          for (int y = 0; y < outDim[1]; y += kernelSizeY) {
            for (int x = 0; x < outDim[0]; x += kernelSizeX) {
              outputData.set(x + xx, y + yy, z, inputData.getData()[index++]);
            }
          }
        }
      }
    }
    inputData.freeRef();
    return outputData;
  }

  @SuppressWarnings("unused")
  public static ImgReshapeLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgReshapeLayer(json);
  }

  public static @SuppressWarnings("unused")
  ImgReshapeLayer[] addRefs(ImgReshapeLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgReshapeLayer::addRef)
        .toArray((x) -> new ImgReshapeLayer[x]);
  }

  public static @SuppressWarnings("unused")
  ImgReshapeLayer[][] addRefs(ImgReshapeLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgReshapeLayer::addRefs)
        .toArray((x) -> new ImgReshapeLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    //assert Arrays.stream(inObj).flatMapToDouble(input-> input.getData().stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    final Result input = inObj[0].addRef();
    ReferenceCounting.freeRefs(inObj);
    final TensorList batch = input.getData();
    @Nonnull final int[] inputDims = batch.getDimensions();
    assert 3 == inputDims.length;
    assert expand || 0 == inputDims[0] % kernelSizeX : (inputDims[0] + " % " + kernelSizeX);
    assert expand || 0 == inputDims[1] % kernelSizeY : (inputDims[1] + " % " + kernelSizeY);
    assert !expand || 0 == inputDims[2] % (kernelSizeX * kernelSizeY);
    //assert input.getData().stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
    Tensor outputDims;
    if (expand) {
      outputDims = new Tensor(inputDims[0] * kernelSizeX, inputDims[1] * kernelSizeY,
          inputDims[2] / (kernelSizeX * kernelSizeY));
    } else {
      outputDims = new Tensor(inputDims[0] / kernelSizeX, inputDims[1] / kernelSizeY,
          inputDims[2] * kernelSizeX * kernelSizeY);
    }
    TensorArray data = new TensorArray(
        RefIntStream.range(0, batch.length()).parallel().mapToObj(RefUtil.wrapInterface(
            (IntFunction<? extends Tensor>) dataIndex -> {
              Tensor inputData = batch.get(dataIndex);
              Tensor temp_43_0002 = expand
                  ? ImgReshapeLayer.copyExpand(inputData == null ? null : inputData.addRef(), outputDims.copy())
                  : ImgReshapeLayer.copyCondense(inputData == null ? null : inputData.addRef(), outputDims.copy());
              if (null != inputData)
                inputData.freeRef();
              return temp_43_0002;
            }, outputDims == null ? null : outputDims.addRef(), batch == null ? null : batch.addRef()))
            .toArray(i -> new Tensor[i]));
    if (null != outputDims)
      outputDims.freeRef();
    if (null != batch)
      batch.freeRef();
    try {
      try {
        return new Result(data, new Result.Accumulator() {
          {
          }

          @Override
          public void accept(DeltaSet<UUID> buffer, TensorList error) {
            //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
            if (input.isAlive()) {
              @Nonnull
              TensorArray tensorArray = new TensorArray(RefIntStream.range(0, error.length()).parallel()
                  .mapToObj(RefUtil.wrapInterface(
                      (IntFunction<? extends Tensor>) dataIndex -> {
                        @Nonnull final Tensor passback = new Tensor(inputDims);
                        @Nullable final Tensor err = error.get(dataIndex);
                        Tensor temp_43_0004 = expand
                            ? ImgReshapeLayer.copyCondense(err == null ? null : err.addRef(),
                            passback == null ? null : passback)
                            : ImgReshapeLayer.copyExpand(err == null ? null : err.addRef(),
                            passback == null ? null : passback.addRef());
                        if (null != err)
                          err.freeRef();
                        return temp_43_0004;
                      }, error == null ? null : error.addRef()))
                  .toArray(i -> new Tensor[i]));
              input.accumulate(buffer == null ? null : buffer.addRef(), tensorArray == null ? null : tensorArray);
            }
            if (null != error)
              error.freeRef();
            if (null != buffer)
              buffer.freeRef();
          }

          public @SuppressWarnings("unused")
          void _free() {
          }
        }) {

          {
          }

          @Override
          public boolean isAlive() {
            return input.isAlive() || !isFrozen();
          }

          public void _free() {
          }
        };
      } finally {
        if (null != data)
          data.freeRef();
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
    json.addProperty("kernelSizeX", kernelSizeX);
    json.addProperty("kernelSizeY", kernelSizeX);
    json.addProperty("expandPlasma", expand);
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return new RefArrayList<>();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  ImgReshapeLayer addRef() {
    return (ImgReshapeLayer) super.addRef();
  }

}
