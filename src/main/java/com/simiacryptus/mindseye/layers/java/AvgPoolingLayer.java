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
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.JsonUtil;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

/**
 * The AvgPoolingLayer class contains a static, final RefConcurrentHashMap field called indexMapCache,
 * as well as an unused, private static final Logger field called log. This class also contains a private
 * int[] field called kernelDims.
 *
 * @docgenVersion 9
 */
@SuppressWarnings("serial")
public class AvgPoolingLayer extends LayerBase {

  /**
   * The constant indexMapCache.
   */
  public static final RefConcurrentHashMap<IndexMapKey, RefMap<Coordinate, RefList<int[]>>> indexMapCache = new RefConcurrentHashMap<>();

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(AvgPoolingLayer.class);
  private int[] kernelDims;

  /**
   * Instantiates a new Avg pooling layer.
   */
  protected AvgPoolingLayer() {
    super();
  }

  /**
   * Instantiates a new Avg pooling layer.
   *
   * @param kernelDims the kernel dims
   */
  public AvgPoolingLayer(@Nonnull final int... kernelDims) {

    this.kernelDims = RefArrays.copyOf(kernelDims, kernelDims.length);
  }

  /**
   * Instantiates a new Avg pooling layer.
   *
   * @param id         the id
   * @param kernelDims the kernel dims
   */
  protected AvgPoolingLayer(@Nonnull final JsonObject id, @Nonnull final int... kernelDims) {
    super(id);
    this.kernelDims = RefArrays.copyOf(kernelDims, kernelDims.length);
  }

  /**
   * @param json The JSON object to deserialize
   * @param rs   A map of character sequences to byte arrays
   * @return A new AvgPoolingLayer
   * @docgenVersion 9
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static AvgPoolingLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new AvgPoolingLayer(json, JsonUtil.getIntArray(json.getAsJsonArray("heapCopy")));
  }

  @RefIgnore
  private static synchronized RefMap<Coordinate, RefList<int[]>> getCoordMap(final int[] kernelDims,
                                                                             final int[] outDims) {
    RefMap<Coordinate, RefList<int[]>> coordinateRefListRefMap;
    synchronized (indexMapCache) {
      coordinateRefListRefMap = indexMapCache.computeIfAbsent(new IndexMapKey(kernelDims, outDims), AvgPoolingLayer::getIndexMap);
    }
    return coordinateRefListRefMap;
//    try {
//    synchronized (AvgPoolingLayer.indexMapCache) {
//      coordinateRefListRefMap = AvgPoolingLayer.indexMapCache.get(new IndexMapKey(kernelDims, outDims));
//    }
//    } catch (@Nonnull final ExecutionException e) {
//      throw new RuntimeException(e);
//    }
  }

  private static RefMap<Coordinate, RefList<int[]>> getIndexMap(@Nonnull IndexMapKey key) {
    Tensor tensor = new Tensor(key.output);
    try {
      final int[] ksize = key.kernel;
      return tensor.coordStream(false).map(x -> x.copy())
          .collect(RefCollectors.toMap(o -> o, o -> {
            @Nonnull Tensor blank = new Tensor(ksize);
            try {
              int[] coords = o.getCoords();
              return blank.coordStream(true).map(kernelCoord -> {
                int[] kernelCoords = kernelCoord.getCoords();
                @Nonnull final int[] r = new int[coords.length];
                for (int i = 0; i < coords.length; i++) {
                  r[i] = coords[i] * ksize[i] + kernelCoords[i];
                }
                return r;
              }).collect(RefCollectors.toList());
            } finally {
              blank.freeRef();
            }
          }));
    } finally {
      tensor.freeRef();
    }
  }

  @Nonnull
  @SuppressWarnings("unchecked")
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final int kernelSize = Tensor.length(kernelDims);
    Result in0 = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    final TensorList data = in0.getData();
    @Nonnull final int[] inputDims = data.getDimensions();
    final int[] newDims = RefIntStream.range(0, inputDims.length).map(i -> {
      assert 0 == inputDims[i] % kernelDims[i] : inputDims[i] + ":" + kernelDims[i];
      return inputDims[i] / kernelDims[i];
    }).toArray();
    final RefMap<Coordinate, RefList<int[]>> coordMap = getCoordMap(kernelDims, newDims);
    TensorArray tensorArray = fwd(kernelSize, data, newDims, coordMap.addRef());
    boolean alive = in0.isAlive();
    final Result.Accumulator accumulator1 = in0.getAccumulator();
    final boolean alive1 = in0.isAlive();
    in0.freeRef();
    Result.Accumulator accumulator = new Accumulator(coordMap, inputDims, kernelSize, accumulator1, alive1);
    return new Result(tensorArray, accumulator, alive);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.add("heapCopy", JsonUtil.getJson(kernelDims));
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  /**
   * This method frees the object.
   *
   * @docgenVersion 9
   */
  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  AvgPoolingLayer addRef() {
    return (AvgPoolingLayer) super.addRef();
  }

  @NotNull
  private TensorArray fwd(int kernelSize, TensorList data, int[] newDims, RefMap<Coordinate, RefList<int[]>> coordMap) {
    return new TensorArray(RefIntStream.range(0, data.length())
        .mapToObj(RefUtil.wrapInterface((IntFunction<Tensor>) dataIndex -> {
          @Nullable final Tensor input = data.get(dataIndex);
          @Nonnull final Tensor output = new Tensor(newDims);
          coordMap.forEach((k, v) -> {
            double sum = v.stream()
                .mapToDouble(RefUtil.wrapInterface(input::get, input.addRef()))
                .sum();
            v.freeRef();
            if (Double.isFinite(sum)) {
              output.add(k, sum / kernelSize);
            }
          });
          input.freeRef();
          return output;
        }, data, coordMap))
        .toArray(Tensor[]::new));
  }

  /**
   * This class represents a key for an index map.
   *
   * @param kernel The Kernel.
   * @param output The Output.
   * @docgenVersion 9
   */
  public static final class IndexMapKey {
    /**
     * The Kernel.
     */
    final int[] kernel;
    /**
     * The Output.
     */
    final int[] output;

    /**
     * Instantiates a new Index map key.
     *
     * @param kernel the kernel
     * @param output the output
     */
    public IndexMapKey(final int[] kernel, final int[] output) {
      super();
      this.kernel = kernel;
      this.output = output;
    }

    /**
     * Instantiates a new Index map key.
     *
     * @param kernel the kernel
     * @param input  the input
     * @param output the output
     */
    public IndexMapKey(@Nonnull final Tensor kernel, @Nullable final Tensor input, @Nonnull final Tensor output) {
      super();
      if (null != input)
        input.freeRef();
      this.kernel = kernel.getDimensions();
      kernel.freeRef();
      this.output = output.getDimensions();
      output.freeRef();
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
      @Nullable final IndexMapKey other = (IndexMapKey) obj;
      return Arrays.equals(kernel, other.kernel) && Arrays.equals(output, other.output);
    }

    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + Arrays.hashCode(kernel);
      result = prime * result + Arrays.hashCode(output);
      return result;
    }
  }

  /**
   * The Accumulator class is used to track the results of a kernel operation.
   *
   * @param RefMap<Coordinate, RefList<int[]>> coordMap A map of Coordinates to integer arrays.
   * @param int[]              inputDims The dimensions of the input array.
   * @param int                kernelSize The size of the kernel.
   * @param Result.Accumulator accumulator An accumulator for tracking the results of the kernel operation.
   * @param boolean            alive A flag to track whether the Accumulator is still needed.
   * @docgenVersion 9
   */
  private static class Accumulator extends Result.Accumulator {

    private final RefMap<Coordinate, RefList<int[]>> coordMap;
    private final int[] inputDims;
    private final int kernelSize;
    private Result.Accumulator accumulator;
    private boolean alive;

    /**
     * Instantiates a new Accumulator.
     *
     * @param coordMap    the coord map
     * @param inputDims   the input dims
     * @param kernelSize  the kernel size
     * @param accumulator the accumulator
     * @param alive       the alive
     */
    public Accumulator(RefMap<Coordinate, RefList<int[]>> coordMap, int[] inputDims, int kernelSize, Result.Accumulator accumulator, boolean alive) {
      this.coordMap = coordMap;
      this.inputDims = inputDims;
      this.kernelSize = kernelSize;
      this.accumulator = accumulator;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
      if (alive) {
        @Nonnull
        TensorArray tensorArray = new TensorArray(RefIntStream.range(0, delta.length())
            .mapToObj(RefUtil.wrapInterface((IntFunction<Tensor>) dataIndex -> {
                  @Nullable
                  Tensor tensor = delta.get(dataIndex);
                  @Nonnull final Tensor backSignal = new Tensor(inputDims);
                  coordMap.forEach((k, v) -> {
                    final double outputValue = tensor.get(k);
                    v.forEach(inputCoord -> backSignal.add(inputCoord, outputValue / kernelSize));
                    v.freeRef();
                  });
                  tensor.freeRef();
                  return backSignal;
                }, delta)
            ).toArray(Tensor[]::new));
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        this.accumulator.accept(buffer1, tensorArray);
      } else {
        delta.freeRef();
      }
      if (null != buffer)
        buffer.freeRef();
    }

    /**
     * Frees resources used by this object.
     *
     * @docgenVersion 9
     */
    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator.freeRef();
      coordMap.freeRef();
    }
  }
}
