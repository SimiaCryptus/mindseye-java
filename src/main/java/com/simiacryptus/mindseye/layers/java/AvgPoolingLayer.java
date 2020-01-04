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

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.util.JsonUtil;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map.Entry;
import java.util.UUID;
import java.util.concurrent.ExecutionException;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware
class AvgPoolingLayer extends LayerBase {

  public static final LoadingCache<AvgPoolingLayer.IndexMapKey, com.simiacryptus.ref.wrappers.RefMap<Coordinate, com.simiacryptus.ref.wrappers.RefList<int[]>>> indexMapCache = CacheBuilder
      .newBuilder().build(new LayerCacheLoader());
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(AvgPoolingLayer.class);
  private int[] kernelDims;

  protected AvgPoolingLayer() {
    super();
  }

  public AvgPoolingLayer(@Nonnull final int... kernelDims) {

    this.kernelDims = com.simiacryptus.ref.wrappers.RefArrays.copyOf(kernelDims, kernelDims.length);
  }

  protected AvgPoolingLayer(@Nonnull final JsonObject id, @Nonnull final int... kernelDims) {
    super(id);
    this.kernelDims = com.simiacryptus.ref.wrappers.RefArrays.copyOf(kernelDims, kernelDims.length);
  }

  @SuppressWarnings("unused")
  public static AvgPoolingLayer fromJson(@Nonnull final JsonObject json,
                                         com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new AvgPoolingLayer(json, JsonUtil.getIntArray(json.getAsJsonArray("heapCopy")));
  }

  public static @SuppressWarnings("unused")
  AvgPoolingLayer[] addRefs(AvgPoolingLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(AvgPoolingLayer::addRef)
        .toArray((x) -> new AvgPoolingLayer[x]);
  }

  public static @SuppressWarnings("unused")
  AvgPoolingLayer[][] addRefs(AvgPoolingLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(AvgPoolingLayer::addRefs)
        .toArray((x) -> new AvgPoolingLayer[x][]);
  }

  private static synchronized com.simiacryptus.ref.wrappers.RefMap<Coordinate, com.simiacryptus.ref.wrappers.RefList<int[]>> getCoordMap(
      final int[] kernelDims, final int[] outDims) {
    try {
      return AvgPoolingLayer.indexMapCache.get(new AvgPoolingLayer.IndexMapKey(kernelDims, outDims));
    } catch (@Nonnull final ExecutionException e) {
      throw new RuntimeException(e);
    }
  }

  @Nonnull
  @SuppressWarnings("unchecked")
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final int kernelSize = Tensor.length(kernelDims);
    final TensorList data = inObj[0].getData();
    @Nonnull final int[] inputDims = data.getDimensions();
    final int[] newDims = com.simiacryptus.ref.wrappers.RefIntStream.range(0, inputDims.length).map(i -> {
      assert 0 == inputDims[i] % kernelDims[i] : inputDims[i] + ":" + kernelDims[i];
      return inputDims[i] / kernelDims[i];
    }).toArray();
    final com.simiacryptus.ref.wrappers.RefMap<Coordinate, com.simiacryptus.ref.wrappers.RefList<int[]>> coordMap = AvgPoolingLayer
        .getCoordMap(kernelDims, newDims);
    final Tensor[] outputValues = com.simiacryptus.ref.wrappers.RefIntStream.range(0, data.length())
        .mapToObj(dataIndex -> {
          @Nullable final Tensor input = data.get(dataIndex);
          @Nonnull final Tensor output = new Tensor(newDims);
          for (@Nonnull final Entry<Coordinate, com.simiacryptus.ref.wrappers.RefList<int[]>> entry : coordMap.entrySet()) {
            double sum = entry.getValue().stream().mapToDouble(inputCoord -> input.get(inputCoord)).sum();
            if (Double.isFinite(sum)) {
              output.add(entry.getKey(), sum / kernelSize);
            }
          }
          return output;
        }).toArray(i -> new Tensor[i]);
    return new Result(new TensorArray(outputValues),
        (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
          if (inObj[0].isAlive()) {
            final Tensor[] passback = com.simiacryptus.ref.wrappers.RefIntStream.range(0, delta.length())
                .mapToObj(dataIndex -> {
                  @Nullable
                  Tensor tensor = delta.get(dataIndex);
                  @Nonnull final Tensor backSignal = new Tensor(inputDims);
                  for (@Nonnull final Entry<Coordinate, com.simiacryptus.ref.wrappers.RefList<int[]>> outputMapping : coordMap
                      .entrySet()) {
                    final double outputValue = tensor.get(outputMapping.getKey());
                    for (@Nonnull final int[] inputCoord : outputMapping.getValue()) {
                      backSignal.add(inputCoord, outputValue / kernelSize);
                    }
                  }
                  return backSignal;
                }).toArray(i -> new Tensor[i]);
            @Nonnull
            TensorArray tensorArray = new TensorArray(passback);
            inObj[0].accumulate(buffer, tensorArray);
          }
        }) {

      @Override
      public boolean isAlive() {
        return inObj[0].isAlive();
      }

      public void _free() {
      }
    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
                            DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.add("heapCopy", JsonUtil.getJson(kernelDims));
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
  AvgPoolingLayer addRef() {
    return (AvgPoolingLayer) super.addRef();
  }

  public static final @com.simiacryptus.ref.lang.RefAware
  class IndexMapKey {
    final int[] kernel;
    final int[] output;

    public IndexMapKey(final int[] kernel, final int[] output) {
      super();
      this.kernel = kernel;
      this.output = output;
    }

    public IndexMapKey(@Nonnull final Tensor kernel, final Tensor input, @Nonnull final Tensor output) {
      super();
      this.kernel = kernel.getDimensions();
      this.output = output.getDimensions();
    }

    @Override
    public boolean equals(@Nullable final Object obj) {
      if (this == obj) {
        return true;
      }
      if (obj == null) {
        return false;
      }
      if (getClass() != obj.getClass()) {
        return false;
      }
      @Nullable final AvgPoolingLayer.IndexMapKey other = (AvgPoolingLayer.IndexMapKey) obj;
      if (!com.simiacryptus.ref.wrappers.RefArrays.equals(kernel, other.kernel)) {
        return false;
      }
      return com.simiacryptus.ref.wrappers.RefArrays.equals(output, other.output);
    }

    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + com.simiacryptus.ref.wrappers.RefArrays.hashCode(kernel);
      result = prime * result + com.simiacryptus.ref.wrappers.RefArrays.hashCode(output);
      return result;
    }
  }

  private static @com.simiacryptus.ref.lang.RefAware
  class LayerCacheLoader extends
      CacheLoader<IndexMapKey, com.simiacryptus.ref.wrappers.RefMap<Coordinate, com.simiacryptus.ref.wrappers.RefList<int[]>>> {
    @Override
    public com.simiacryptus.ref.wrappers.RefMap<Coordinate, com.simiacryptus.ref.wrappers.RefList<int[]>> load(
        @NotNull final IndexMapKey key) {
      final int[] ksize = key.kernel;
      Tensor tensor = new Tensor(key.output);
      return tensor.coordStream(true).collect(com.simiacryptus.ref.wrappers.RefCollectors.toMap(o -> o, o -> {
        @Nonnull
        Tensor blank = new Tensor(ksize);
        return blank.coordStream(true).map(kernelCoord -> {
          int[] coords = o.getCoords();
          @Nonnull final int[] r = new int[coords.length];
          for (int i = 0; i < coords.length; i++) {
            r[i] = coords[i] * ksize[i] + kernelCoord.getCoords()[i];
          }
          return r;
        }).collect(com.simiacryptus.ref.wrappers.RefCollectors.toList());
      }));
    }
  }
}
