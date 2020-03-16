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
import com.simiacryptus.ref.wrappers.*;
import org.jetbrains.annotations.NotNull;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public class PhotoUnpoolingLayer extends LayerBase {

  public PhotoUnpoolingLayer() {
    super();
  }

  protected PhotoUnpoolingLayer(@Nonnull final JsonObject json) {
    super(json);
  }

  @Nonnull
  public static Tensor copyCondense(@Nonnull final Tensor inputData, @Nonnull final Tensor outputData,
                                    @Nonnull Tensor referenceData) {
    @Nonnull final int[] inDim = inputData.getDimensions();
    @Nonnull final int[] outDim = outputData.getDimensions();
    assert 3 == inDim.length;
    assert 3 == outDim.length;
    assert inDim[0] >= outDim[0];
    assert inDim[1] >= outDim[1];
    assert inDim[2] == outDim[2];
    assert 0 == inDim[0] % outDim[0];
    assert 0 == inDim[1] % outDim[1];
    final int kernelSizeX = inDim[0] / outDim[0];
    final int kernelSizeY = inDim[0] / outDim[0];
    assert RefArrays.equals(referenceData.getDimensions(), inDim);
    final int[] referenceDataDimensions = referenceData.getDimensions();
    for (int z = 0; z < inDim[2]; z++) {
      for (int y = 0; y < inDim[1]; y += kernelSizeY) {
        for (int x = 0; x < inDim[0]; x += kernelSizeX) {

          int xx = -1;
          int yy = -1;
          double maxV = Double.NaN;
          for (int xxx = 0; xxx < kernelSizeX; xxx++) {
            for (int yyy = 0; yyy < kernelSizeY; yyy++) {
              final double thisV = referenceData.get((x + xxx) % referenceDataDimensions[0],
                  (y + yyy) % referenceDataDimensions[1], z);
              if (Double.isNaN(maxV) || thisV > maxV) {
                maxV = thisV;
                xx = xxx;
                yy = yyy;
              }
            }
          }
          final double value = inputData.get(x + xx, y + yy, z);
          outputData.set(x / kernelSizeX, y / kernelSizeY, z, value);
        }
      }
    }
    referenceData.freeRef();
    inputData.freeRef();
    return outputData;
  }

  @Nonnull
  public static Tensor copyExpand(@Nonnull final Tensor inputData, @Nonnull final Tensor outputData,
                                  @Nonnull Tensor referenceData) {
    @Nonnull final int[] inDim = inputData.getDimensions();
    @Nonnull final int[] outDim = outputData.getDimensions();
    assert 3 == inDim.length;
    assert 3 == outDim.length;
    assert inDim[0] <= outDim[0];
    assert inDim[1] <= outDim[1];
    assert inDim[2] == outDim[2];
    assert RefArrays.equals(referenceData.getDimensions(), outDim) : RefString.format("%s != %s",
        RefArrays.toString(referenceData.getDimensions()), RefArrays.toString(outDim));
    final int kernelSizeX = outDim[0] / inDim[0];
    final int kernelSizeY = outDim[0] / inDim[0];
    final int[] referenceDataDimensions = referenceData.getDimensions();
    for (int z = 0; z < outDim[2]; z++) {
      for (int y = 0; y < outDim[1]; y += kernelSizeY) {
        for (int x = 0; x < outDim[0]; x += kernelSizeX) {
          final double value = inputData.get(x / kernelSizeX, y / kernelSizeY, z);
          int xx = -1;
          int yy = -1;
          double maxV = Double.NaN;
          for (int xxx = 0; xxx < kernelSizeX; xxx++) {
            for (int yyy = 0; yyy < kernelSizeY; yyy++) {
              final double thisV = referenceData.get((x + xxx) % referenceDataDimensions[0],
                  (y + yyy) % referenceDataDimensions[1], z);
              if (Double.isNaN(maxV) || thisV > maxV) {
                maxV = thisV;
                xx = xxx;
                yy = yyy;
              }
            }
          }
          outputData.set((x + xx) % referenceDataDimensions[0], (y + yy) % referenceDataDimensions[1], z, value);
        }
      }
    }
    referenceData.freeRef();
    inputData.freeRef();
    return outputData;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static PhotoUnpoolingLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new PhotoUnpoolingLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    //assert Arrays.stream(inObj).flatMapToDouble(input-> input.getData().stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    final Result input = inObj[0].addRef();
    final TensorList batch = input.getData();
    final TensorList referencebatch = inObj[1].getData();
    RefUtil.freeRef(inObj);
    @Nonnull final int[] inputDims = batch.getDimensions();
    assert 3 == inputDims.length;
    TensorArray data = fwd(batch, referencebatch.addRef());
    boolean alive = input.isAlive();
    Result.Accumulator accumulator = new Accumulator(referencebatch, inputDims, input.getAccumulator(), input.isAlive());
    input.freeRef();
    return new Result(data, accumulator, alive);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return new RefArrayList<>();
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  PhotoUnpoolingLayer addRef() {
    return (PhotoUnpoolingLayer) super.addRef();
  }

  @NotNull
  private TensorArray fwd(TensorList batch, TensorList referencebatch) {
    Tensor outputDims = new Tensor(referencebatch.getDimensions());
    return new TensorArray(RefIntStream.range(0, batch.length()).parallel()
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
          Tensor inputData = batch.get(dataIndex);
          Tensor referenceData = referencebatch.get(dataIndex);
          Tensor temp_34_0003 = PhotoUnpoolingLayer.copyExpand(inputData.addRef(),
              outputDims.copy(), referenceData.addRef());
          referenceData.freeRef();
          inputData.freeRef();
          return temp_34_0003;
        }, outputDims, referencebatch, batch))
        .toArray(Tensor[]::new));
  }

  private static class Accumulator extends Result.Accumulator {

    private final TensorList referencebatch;
    private final int[] inputDims;
    private Result.Accumulator accumulator;
    private boolean alive;

    public Accumulator(TensorList referencebatch, int[] inputDims, Result.Accumulator accumulator, boolean alive) {
      this.referencebatch = referencebatch;
      this.inputDims = inputDims;
      this.accumulator = accumulator;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList error) {
      //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
      if (alive) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(RefIntStream.range(0, error.length()).parallel()
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
              @Nonnull final Tensor passback = new Tensor(inputDims);
              @Nullable final Tensor err = error.get(dataIndex);
              Tensor referenceData = referencebatch.get(dataIndex);
              Tensor temp_34_0005 = PhotoUnpoolingLayer.copyCondense(err.addRef(),
                  passback, referenceData.addRef());
              referenceData.freeRef();
              err.freeRef();
              return temp_34_0005;
            }, referencebatch.addRef(), error.addRef()))
            .toArray(Tensor[]::new));
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        this.accumulator.accept(buffer1, tensorArray);
      }
      error.freeRef();
      if (null != buffer)
        buffer.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      accumulator.freeRef();
      referencebatch.freeRef();
      super._free();
    }
  }
}
