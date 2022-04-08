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
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * CoordinateAssemblyLayer class.
 *
 * @author Author
 * @docgenVersion 9
 */
@SuppressWarnings("serial")
public class CoordinateAssemblyLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(CoordinateAssemblyLayer.class);
  private final boolean discardCoordinates;

  public CoordinateAssemblyLayer() {
    this(true);
  }

  public CoordinateAssemblyLayer(boolean discardCoordinates) {
    super();
    this.discardCoordinates = discardCoordinates;
  }

  /**
   * Instantiates a new Img band bias layer.
   *
   * @param json the json
   */
  protected CoordinateAssemblyLayer(@Nonnull final JsonObject json) {
    super(json);
    discardCoordinates = (json.getAsJsonPrimitive("discardCoordinates")).getAsBoolean();
  }

  /**
   * Creates a new CoordinateAssemblyLayer from the given JSON object.
   *
   * @param json the JSON object to create the layer from
   * @param rs   the map of resources to use
   * @return the new CoordinateAssemblyLayer
   * @docgenVersion 9
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static CoordinateAssemblyLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new CoordinateAssemblyLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nullable final Result... inObj) {
    assert inObj != null;
    Result result = eval(inObj[0].addRef(), inObj[1].addRef());
    RefUtil.freeRef(inObj);
    return result;
  }

  /**
   * @param input      the input to be evaluated
   * @param inputModel the input model to be used
   * @return the result of the evaluation
   * @throws NullPointerException if either input or inputModel is null
   * @docgenVersion 9
   */
  @Nonnull
  public Result eval(@Nonnull final Result input, @Nonnull final Result inputModel) {
    try {
      TensorList inputData = input.getData();
      int[] dimensions = Result.getDimensions(inputModel.addRef());
      if ((dimensions[0] * dimensions[1]) != inputData.length()) {
        log.info(String.format("Mismatching data length: %d != %d", Arrays.toString(dimensions), inputData.length()));
      }
      TensorArray data = fwd(dimensions, inputData);
      boolean alive1 = input.isAlive();
      boolean alive2 = inputModel.isAlive();
      Accumulator accumulator = new Accumulator(dimensions, input.getAccumulator(), alive1, inputModel.getAccumulator(), alive2, inputData.length(), discardCoordinates);
      return new Result(data, accumulator, alive1 || !isFrozen());
    } finally {
      input.freeRef();
      inputModel.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("discardCoordinates", discardCoordinates);
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefList.empty();
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
  CoordinateAssemblyLayer addRef() {
    return (CoordinateAssemblyLayer) super.addRef();
  }

  @NotNull
  private TensorArray fwd(int[] dimensions, TensorList inputData) {
    Tensor coordImage = new Tensor(dimensions[0], dimensions[1]);
    try {
      // Use modulus below to help standardized unit tests function
      int coordOffset = discardCoordinates ? 2 : 0;
      Tensor tensor = new Tensor(IntStream.range(0, dimensions[2]).mapToObj(x -> x)
          .flatMapToDouble(z -> DoubleStream.of(coordImage.coordStream(false).mapToDouble(c ->
              inputData.get(c.getIndex() % inputData.length(), coordOffset + z)).toArray())).toArray(), dimensions);

      return new TensorArray(tensor);
    } finally {
      inputData.freeRef();
      coordImage.freeRef();
    }
  }

  /**
   * The Accumulator class is used to hold data about an accumulator.
   *
   * @param dimensions         An array of integers representing the dimensions of the accumulator.
   * @param inputDataLength    An integer representing the length of the input data.
   * @param accumulator2       A Result.Accumulator representing the second accumulator.
   * @param alive2             A boolean representing whether or not the second accumulator is alive.
   * @param discardCoordinates A boolean representing whether or not to discard coordinates.
   * @param accumulator1       A Result.Accumulator representing the first accumulator.
   * @param alive1             A boolean representing whether or not the first accumulator is alive.
   * @docgenVersion 9
   */
  private static class Accumulator extends Result.Accumulator {

    private final int[] dimensions;
    private final int inputDataLength;
    private final Result.Accumulator accumulator2;
    private final boolean alive2;
    private final boolean discardCoordinates;
    private Result.Accumulator accumulator1;
    private boolean alive1;

    /**
     * Instantiates a new Accumulator.
     *
     * @param dimensions      input resolution
     * @param accumulator1    the accumulator
     * @param alive1          the alive
     * @param inputDataLength Helper parameter to track number of input pixels; used for workaround to help standard unit test function
     */
    public Accumulator(int[] dimensions, Result.Accumulator accumulator1, boolean alive1, Result.Accumulator accumulator2, boolean alive2, int inputDataLength, boolean discardCoordinates) {
      this.dimensions = dimensions;
      this.inputDataLength = inputDataLength;
      this.accumulator1 = accumulator1;
      this.accumulator2 = accumulator2;
      this.alive1 = alive1;
      this.alive2 = alive2;
      this.discardCoordinates = discardCoordinates;
    }

    @Override
    public void accept(@Nonnull DeltaSet<UUID> buffer, @Nonnull TensorList data) {
      if (alive1) {
        Tensor image = data.get(0);
        Tensor[] pixels = RefIntStream.range(0, dimensions[1]).mapToObj(x -> x).flatMap(y ->
            RefIntStream.range(0, dimensions[0]).mapToObj(x -> {
              Tensor tensor = new Tensor(1, 1, (discardCoordinates ? 2 : 0) + dimensions[2]);
              try {
                return tensor.mapIndex((v, i) -> {
                  if (discardCoordinates) {
                    if (i == 0) {
                      return 0;
                    } else if (i == 1) {
                      return 0;
                    } else {
                      return image.get(x, y, i - 2);
                    }
                  } else {
                    return image.get(x, y, i);
                  }
                });
              } finally {
                tensor.freeRef();
              }
            })).toArray(Tensor[]::new);
        if (this.inputDataLength != pixels.length) {
          log.info(String.format("Correcting mismatching data length: %d != %d", inputDataLength, pixels.length));
          pixels = RefArrays.copyOfRange(pixels, 0, this.inputDataLength);
        }
        image.freeRef();
        data.freeRef();
        if (alive2) {
          Tensor nullPassbackImage = new Tensor(dimensions);
          nullPassbackImage.fill(0.0);
          accumulator2.accept(buffer.addRef(), new TensorArray(nullPassbackImage));
        }
        accumulator1.accept(buffer, new TensorArray(pixels));
      } else {
        data.freeRef();
        if (alive2) {
          accumulator2.accept(buffer, new TensorArray(new Tensor(dimensions)));
        } else {
          buffer.freeRef();
        }
      }
    }

    /**
     * Frees resources used by this object.
     *
     * @docgenVersion 9
     */
    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator1.freeRef();
      accumulator2.freeRef();
    }
  }
}
