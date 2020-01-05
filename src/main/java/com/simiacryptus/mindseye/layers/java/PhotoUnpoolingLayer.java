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
import com.simiacryptus.ref.wrappers.RefArrayList;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public @RefAware
class PhotoUnpoolingLayer extends LayerBase {

  public PhotoUnpoolingLayer(final int sizeX, final int sizeY) {
    super();
  }

  protected PhotoUnpoolingLayer(@Nonnull final JsonObject json) {
    super(json);
  }

  @Nonnull
  public static Tensor copyCondense(@Nonnull final Tensor inputData, @Nonnull final Tensor outputData,
                                    Tensor referenceData) {
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
    return outputData;
  }

  @Nonnull
  public static Tensor copyExpand(@Nonnull final Tensor inputData, @Nonnull final Tensor outputData,
                                  Tensor referenceData) {
    @Nonnull final int[] inDim = inputData.getDimensions();
    @Nonnull final int[] outDim = outputData.getDimensions();
    assert 3 == inDim.length;
    assert 3 == outDim.length;
    assert inDim[0] <= outDim[0];
    assert inDim[1] <= outDim[1];
    assert inDim[2] == outDim[2];
    assert RefArrays.equals(referenceData.getDimensions(), outDim) : String.format(
        "%s != %s", RefArrays.toString(referenceData.getDimensions()),
        RefArrays.toString(outDim));
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
    return outputData;
  }

  @SuppressWarnings("unused")
  public static PhotoUnpoolingLayer fromJson(@Nonnull final JsonObject json,
                                             Map<CharSequence, byte[]> rs) {
    return new PhotoUnpoolingLayer(json);
  }

  public static @SuppressWarnings("unused")
  PhotoUnpoolingLayer[] addRefs(PhotoUnpoolingLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(PhotoUnpoolingLayer::addRef)
        .toArray((x) -> new PhotoUnpoolingLayer[x]);
  }

  public static @SuppressWarnings("unused")
  PhotoUnpoolingLayer[][] addRefs(PhotoUnpoolingLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(PhotoUnpoolingLayer::addRefs)
        .toArray((x) -> new PhotoUnpoolingLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    //assert Arrays.stream(inObj).flatMapToDouble(input-> input.getData().stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    final Result input = inObj[0];
    final TensorList batch = input.getData();
    final TensorList referencebatch = inObj[1].getData();
    @Nonnull final int[] inputDims = batch.getDimensions();
    assert 3 == inputDims.length;
    Tensor outputDims;
    outputDims = new Tensor(inObj[1].getData().getDimensions());
    TensorArray data = new TensorArray(
        RefIntStream.range(0, batch.length()).parallel().mapToObj(dataIndex -> {
          Tensor inputData = batch.get(dataIndex);
          Tensor referenceData = referencebatch.get(dataIndex);
          return PhotoUnpoolingLayer.copyExpand(inputData, outputDims.copy(), referenceData);
        }).toArray(i -> new Tensor[i]));
    return new Result(data, (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList error) -> {
      //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
      if (input.isAlive()) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(
            RefIntStream.range(0, error.length()).parallel().mapToObj(dataIndex -> {
              @Nonnull final Tensor passback = new Tensor(inputDims);
              @Nullable final Tensor err = error.get(dataIndex);
              Tensor referenceData = referencebatch.get(dataIndex);
              return PhotoUnpoolingLayer.copyCondense(err, passback, referenceData);
            }).toArray(i -> new Tensor[i]));
        input.accumulate(buffer, tensorArray);
      }
    }) {

      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
      }

      public void _free() {
      }
    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources,
                            DataSerializer dataSerializer) {
    return super.getJsonStub();
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
  PhotoUnpoolingLayer addRef() {
    return (PhotoUnpoolingLayer) super.addRef();
  }

}
